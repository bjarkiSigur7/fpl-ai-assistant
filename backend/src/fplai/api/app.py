"""FastAPI application serving predictions, plans and recommendations to the dashboard.

Run locally (from ``backend/``)::

    uv run uvicorn fplai.api.app:app --reload --port 8000

All routes live under ``/api`` and return the pydantic models from
:mod:`fplai.api.schemas` — which import the optimizer contracts
(:class:`~fplai.optimizer.plans.Recommendation`, :class:`~fplai.optimizer.plans.DreamTeam`,
:class:`~fplai.optimizer.state.SquadState`) verbatim, so the OpenAPI schema is the
frontend's source of truth.  Design rules:

* startup is lightweight — no model artifacts or LightGBM/HiGHS imports at import time;
* every parquet/JSON read goes through :data:`fplai.api.cache.cache`, an mtime-aware
  loader (the refresh pipeline rewrites files under the API's feet);
* expensive work (refresh pipeline, my-team MILP solves) runs in single-flight
  background threads (:mod:`fplai.api.jobs`) — no request ever blocks on the solver;
* errors use FastAPI's standard ``{"detail": ...}`` shape.

Pre-launch vs live: while the FPL API still serves the finished 2025-26 season the app
runs in demo mode off the 2025-26 backtest artifacts; ``GET /api/state`` exposes the
``pre_launch`` flag and the 2026-27 GW1 deadline countdown the dashboard shows.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fplai import config
from fplai.api import jobs
from fplai.api.cache import cache
from fplai.api.schemas import (
    ChipCurvePoint,
    ChipCurvesResponse,
    DataFreshness,
    FixturePrediction,
    HealthResponse,
    JobStatusResponse,
    MyTeamResponse,
    PlayerDetail,
    PlayerGwHistoryRow,
    PlayerIdentity,
    PlayerPrediction,
    PredictionsResponse,
    RefreshStatusResponse,
    SeasonStateModel,
    SnapshotFreshness,
    StateResponse,
)
from fplai.optimizer.plans import DreamTeam, Recommendation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pandas as pd

API_VERSION = "0.1.0"

#: Announced 2026-27 GW1 deadline (UTC) — the pre-launch countdown target until the
#: FPL API goes live and starts serving real event deadlines.
GW1_DEADLINE_2026_27 = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)

#: Serve a cached my-team recommendation only while younger than this (task contract).
RECOMMENDATION_TTL_S = 6 * 3600.0

_SORT_FIELDS = (
    "xp",
    "price",
    "web_name",
    "position",
    "n_fixtures",
    "q0",
    "xp_appearance",
    "xp_goals",
    "xp_assists",
    "xp_cs",
    "xp_concede",
    "xp_saves",
    "xp_defcon",
    "xp_bonus",
    "xp_cards",
    "xp_other",
)
SortField = Literal[_SORT_FIELDS]  # type: ignore[valid-type]
Position = Literal["GKP", "DEF", "MID", "FWD"]


# --------------------------------------------------------------------------------------
# Small helpers (paths resolved lazily so tests can monkeypatch fplai.config)
# --------------------------------------------------------------------------------------


def _processed(name: str) -> Path:
    return config.PROCESSED_DIR / name


def _load_parquet_or_404(name: str, hint: str) -> pd.DataFrame:
    try:
        return cache.load_parquet(_processed(name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{name} not found — {hint}") from exc


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (NaN/NA -> None, numpy scalars -> python)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _team_names(season: int) -> tuple[dict[int, str], dict[int, str]]:
    """``team_code -> (name, short_name)`` maps for one season (best-effort)."""
    try:
        teams = cache.load_parquet(_processed("teams.parquet"))
    except FileNotFoundError:
        return {}, {}
    rows = teams[teams["season"] == season]
    names = {int(c): str(n) for c, n in zip(rows["team_code"], rows["name"], strict=True)}
    shorts = {
        int(c): str(s) for c, s in zip(rows["team_code"], rows["short_name"], strict=True)
    }
    return names, shorts


def _fixture_kickoffs() -> dict[tuple[int, int], str]:
    """``(season, fpl_fixture_id) -> kickoff ISO string`` from fixtures.parquet (best-effort)."""
    try:
        fx = cache.load_parquet(_processed("fixtures.parquet"))
    except FileNotFoundError:
        return {}
    out: dict[tuple[int, int], str] = {}
    for season, fid, kick in zip(
        fx["season"], fx["fpl_fixture_id"], fx["kickoff_utc"], strict=True
    ):
        if kick is not None and kick == kick:  # NaT-safe
            out[(int(season), int(fid))] = str(kick.isoformat())
    return out


def _fixture_predictions(
    pred: pd.DataFrame, shorts: dict[int, str], kickoffs: dict[tuple[int, int], str] | None = None
) -> list[FixturePrediction]:
    """Per-fixture prediction rows -> models, with opponent short names + kickoffs attached.

    Pass a precomputed ``_fixture_kickoffs()`` dict when calling in a loop.
    """
    if kickoffs is None:
        kickoffs = _fixture_kickoffs()
    out: list[FixturePrediction] = []
    for rec in _records(pred):
        opp = rec.get("opponent_code")
        rec["opponent_short_name"] = shorts.get(int(opp)) if opp is not None else None
        rec["kickoff_utc"] = kickoffs.get((int(rec["season"]), int(rec["fpl_fixture_id"])))
        out.append(FixturePrediction.model_validate(rec))
    return out


# --------------------------------------------------------------------------------------
# App + router
# --------------------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lightweight startup: record where data lives; never import model artifacts."""
    app.state.processed_dir = str(config.PROCESSED_DIR)
    app.state.models_dir = str(config.MODELS_DIR)
    yield


app = FastAPI(
    title="FPL AI Assistant",
    version=API_VERSION,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------------------
# /health and /state
# --------------------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(version=API_VERSION, time_utc=datetime.now(UTC))


def _latest_snapshot_dir() -> Path | None:
    root = config.RAW_DIR / "fpl_api" / "snapshots"
    if not root.is_dir():
        return None
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    return dirs[-1] if dirs else None


def _season_state_from_disk(snap_dir: Path | None) -> SeasonStateModel | None:
    """Parse SeasonState from the latest snapshot's bootstrap.json (no network)."""
    if snap_dir is None:
        return None
    bootstrap_path = snap_dir / "bootstrap.json"
    if not bootstrap_path.exists():
        return None
    from fplai.data.fpl_api import parse_season_state

    try:
        state = parse_season_state(cache.load_json(bootstrap_path))
    except (ValueError, json.JSONDecodeError):
        return None
    return SeasonStateModel(
        season=state.season,
        is_live_2026_27=state.is_live_2026_27,
        current_gw=state.current_gw,
        next_gw=state.next_gw,
        next_deadline_utc=state.next_deadline_utc,
        total_players=state.total_players,
        static_content_url=state.static_content_url,
    )


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    """Season state (from the latest on-disk snapshot), data freshness and manifest."""
    now = datetime.now(UTC)
    snap_dir = _latest_snapshot_dir()
    season_state = _season_state_from_disk(snap_dir)

    latest_snapshot: SnapshotFreshness | None = None
    if snap_dir is not None:
        bootstrap_path = snap_dir / "bootstrap.json"
        fixtures_path = snap_dir / "fixtures.json"
        latest_snapshot = SnapshotFreshness(
            date=snap_dir.name,
            bootstrap_mtime_utc=_mtime_utc(bootstrap_path) if bootstrap_path.exists() else None,
            fixtures_mtime_utc=_mtime_utc(fixtures_path) if fixtures_path.exists() else None,
        )

    processed: dict[str, datetime] = {}
    if config.PROCESSED_DIR.is_dir():
        for path in sorted(config.PROCESSED_DIR.iterdir()):
            if path.suffix in (".parquet", ".json"):
                processed[path.name] = _mtime_utc(path)

    manifest: dict[str, Any] | None = None
    models_trained: datetime | None = None
    try:
        manifest = cache.load_json(config.MODELS_DIR / "manifest.json")
        created = manifest.get("created_utc") if isinstance(manifest, dict) else None
        if created:
            models_trained = datetime.fromisoformat(str(created))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        manifest = None

    predictions_season: int | None = None
    predictions_gws: list[int] = []
    try:
        pred_gw = cache.load_parquet(_processed("predictions_gw.parquet"))
        if len(pred_gw):
            predictions_season = int(pred_gw["season"].max())
            predictions_gws = sorted(
                int(g)
                for g in pred_gw.loc[pred_gw["season"] == predictions_season, "gw"].unique()
            )
    except FileNotFoundError:
        pass

    pre_launch = not (season_state is not None and season_state.is_live_2026_27)
    if season_state is not None and season_state.next_deadline_utc is not None:
        deadline: datetime | None = season_state.next_deadline_utc
    elif pre_launch:
        deadline = GW1_DEADLINE_2026_27
    else:
        deadline = None
    seconds_left = (deadline - now).total_seconds() if deadline is not None else None

    return StateResponse(
        server_time_utc=now,
        season_state=season_state,
        pre_launch=pre_launch,
        gw1_deadline_utc=deadline,
        seconds_to_gw1_deadline=seconds_left,
        days_to_gw1_deadline=seconds_left / 86400.0 if seconds_left is not None else None,
        freshness=DataFreshness(
            latest_snapshot=latest_snapshot,
            processed=processed,
            models_trained_utc=models_trained,
            predictions_season=predictions_season,
            predictions_gws=predictions_gws,
        ),
        model_manifest=manifest,
    )


# --------------------------------------------------------------------------------------
# /predictions
# --------------------------------------------------------------------------------------


@router.get("/predictions", response_model=PredictionsResponse)
def predictions(
    season: int | None = Query(default=None, ge=2016, le=2026),
    gw: int | None = Query(default=None, ge=1, le=38),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    position: Position | None = None,
    sort: SortField = "xp",
    order: Literal["asc", "desc"] = "desc",
) -> PredictionsResponse:
    """Paginated per-player xP (+ components + minutes q's) for one (season, gw).

    Defaults to the latest season and its earliest GW in ``predictions_gw.parquet``.
    Names/teams/prices are joined from players/teams/player_gw; the per-fixture
    breakdown (DGW legs, ``q0``/``q1``/``q2``/``mu1``/``mu2``) comes from
    ``predictions.parquet``.
    """
    gw_df = _load_parquet_or_404(
        "predictions_gw.parquet", "run `fplai predict` (or `fplai refresh`) first"
    )
    if season is None:
        season = int(gw_df["season"].max())
    in_season = gw_df[gw_df["season"] == season]
    if in_season.empty:
        available = sorted(int(s) for s in gw_df["season"].unique())
        raise HTTPException(
            status_code=404,
            detail=f"no predictions for season {season} (available: {available})",
        )
    if gw is None:
        gw = int(in_season["gw"].min())
    rows = in_season[in_season["gw"] == gw].copy()
    if rows.empty:
        available = sorted(int(g) for g in in_season["gw"].unique())
        raise HTTPException(
            status_code=404,
            detail=f"no predictions for season {season} GW{gw} (available GWs: {available})",
        )

    # Joins: web_name from players.parquet, price from player_gw latest value (fallbacks
    # for older prediction files), team names from teams.parquet.
    if "web_name" not in rows.columns:
        rows["web_name"] = None
    try:
        players = cache.load_parquet(_processed("players.parquet"))
        name_of = dict(
            zip(players["player_code"], players["web_name"], strict=True)
        )
        rows["web_name"] = rows["web_name"].fillna(rows["player_code"].map(name_of))
    except FileNotFoundError:
        pass
    if "price" in rows.columns and rows["price"].isna().any():
        try:
            pgw = cache.load_parquet(_processed("player_gw.parquet"))
            latest_value = (
                pgw.sort_values(["player_code", "season", "gw"], kind="stable")
                .groupby("player_code")["value"]
                .last()
            )
            rows["price"] = rows["price"].fillna(rows["player_code"].map(latest_value))
        except FileNotFoundError:
            pass
    names, shorts = _team_names(season)

    if position is not None:
        rows = rows[rows["position"] == position]
    total = int(len(rows))
    ascending = order == "asc"
    rows = rows.sort_values(
        [sort, "player_code"], ascending=[ascending, True], na_position="last", kind="stable"
    )
    page = rows.iloc[offset : offset + limit]

    fixtures_by_player: dict[int, list[FixturePrediction]] = {}
    try:
        pred = cache.load_parquet(_processed("predictions.parquet"))
        page_codes = set(int(c) for c in page["player_code"])
        sub = pred[
            (pred["season"] == season)
            & (pred["gw"] == gw)
            & pred["player_code"].isin(page_codes)
        ].sort_values(["player_code", "fpl_fixture_id"], kind="stable")
        kickoffs = _fixture_kickoffs()
        for code, grp in sub.groupby("player_code"):
            fixtures_by_player[int(code)] = _fixture_predictions(grp, shorts, kickoffs)
    except FileNotFoundError:
        pass

    players_out: list[PlayerPrediction] = []
    for rec in _records(page):
        code = int(rec["player_code"])
        team_code = rec.get("team_code")
        rec["team_name"] = names.get(int(team_code)) if team_code is not None else None
        rec["team_short_name"] = shorts.get(int(team_code)) if team_code is not None else None
        rec["fixtures"] = fixtures_by_player.get(code, [])
        players_out.append(PlayerPrediction.model_validate(rec))

    return PredictionsResponse(
        season=season,
        gw=gw,
        gws=sorted(int(g) for g in in_season["gw"].unique()),
        total=total,
        limit=limit,
        offset=offset,
        sort=sort,
        order=order,
        players=players_out,
    )


# --------------------------------------------------------------------------------------
# /dream-team and /recommendation (artifact-serving endpoints)
# --------------------------------------------------------------------------------------


def _unwrap(data: Any, key: str) -> Any:
    """Accept both a bare model dump and a ``{key: {...}}`` wrapper."""
    if isinstance(data, dict) and key in data and isinstance(data[key], dict):
        return data[key]
    return data


@router.get("/dream-team", response_model=DreamTeam)
def dream_team(
    season: int | None = Query(default=None, ge=2016, le=2026),
    gw: int | None = Query(default=None, ge=1, le=38),
) -> DreamTeam:
    """The fresh-£100m benchmark squad from ``data/processed/dream_team.json``.

    404 when the file is missing, older than the predictions it derives from, or does
    not match the requested ``season``/``gw``.
    """
    path = _processed("dream_team.json")
    try:
        data = cache.load_json(path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="dream_team.json not found — run `fplai optimize` (or `fplai refresh`) first",
        ) from exc
    dream = DreamTeam.model_validate(_unwrap(data, "dream_team"))

    pred_path = _processed("predictions_gw.parquet")
    if pred_path.exists() and pred_path.stat().st_mtime > path.stat().st_mtime:
        raise HTTPException(
            status_code=404,
            detail="dream_team.json is stale (predictions_gw.parquet is newer) — "
            "re-run `fplai optimize`",
        )
    if season is not None and dream.season != season:
        raise HTTPException(
            status_code=404,
            detail=f"dream team on disk is for season {dream.season}, not {season}",
        )
    if gw is not None and dream.gw != gw:
        raise HTTPException(
            status_code=404,
            detail=f"dream team on disk is for GW{dream.gw}, not GW{gw}",
        )
    return dream


@router.get("/recommendation", response_model=Recommendation)
def recommendation() -> Recommendation:
    """The precomputed weekly verdict from ``data/processed/recommendation.json``."""
    try:
        data = cache.load_json(_processed("recommendation.json"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="recommendation.json not found — run `fplai optimize` "
            "(or `fplai refresh`) first",
        ) from exc
    return Recommendation.model_validate(_unwrap(data, "recommendation"))


# --------------------------------------------------------------------------------------
# /chip-curves
# --------------------------------------------------------------------------------------


@router.get("/chip-curves", response_model=ChipCurvesResponse)
def chip_curves() -> ChipCurvesResponse:
    """Chip EV curves (forced-chip re-solves) from ``chip_curves.parquet``."""
    path = _processed("chip_curves.parquet")
    try:
        df = cache.load_parquet(path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="chip_curves.parquet not found — run `fplai optimize` "
            "(or `fplai refresh`) first",
        ) from exc
    curves = [ChipCurvePoint.model_validate(rec) for rec in _records(df)]
    return ChipCurvesResponse(generated_at=_mtime_utc(path), curves=curves)


# --------------------------------------------------------------------------------------
# /players/{player_code}
# --------------------------------------------------------------------------------------


@router.get("/players/{player_code}", response_model=PlayerDetail)
def player_detail(
    player_code: int,
    season: int | None = Query(default=None, ge=2016, le=2026),
) -> PlayerDetail:
    """Identity, per-GW season history and upcoming per-fixture xP for one player."""
    players = _load_parquet_or_404("players.parquet", "run `fplai build` first")
    match = players[players["player_code"] == player_code]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"unknown player_code {player_code}")
    identity = PlayerIdentity.model_validate(_records(match)[0])

    import pandas as pd

    history_rows: list[PlayerGwHistoryRow] = []
    seasons: list[int] = []
    team_code: int | None = None
    pos: str | None = None
    price: int | None = None
    try:
        pgw = cache.load_parquet(_processed("player_gw.parquet"))
        mine = pgw[pgw["player_code"] == player_code]
        seasons = sorted(int(s) for s in mine["season"].unique())
        if seasons:
            target_season = season if season is not None else seasons[-1]
            if target_season not in seasons:
                raise HTTPException(
                    status_code=404,
                    detail=f"player {player_code} has no history in season "
                    f"{target_season} (available: {seasons})",
                )
            hist = mine[mine["season"] == target_season].sort_values("gw", kind="stable")
            history_rows = [PlayerGwHistoryRow.model_validate(r) for r in _records(hist)]
            latest = mine.sort_values(["season", "gw"], kind="stable").iloc[-1]
            team_code = int(latest["team_code"])
            pos = str(latest["position"]) if pd.notna(latest["position"]) else None
            price = int(latest["value"])
            season = target_season
    except FileNotFoundError:
        pass

    upcoming: list[FixturePrediction] = []
    try:
        pred = cache.load_parquet(_processed("predictions.parquet"))
        mine_pred = pred[pred["player_code"] == player_code].sort_values(
            ["season", "gw", "fpl_fixture_id"], kind="stable"
        )
        if not mine_pred.empty:
            _, shorts = _team_names(int(mine_pred["season"].max()))
            upcoming = _fixture_predictions(mine_pred, shorts)
            if team_code is None:
                last = mine_pred.iloc[-1]
                team_code = int(last["team_code"])
                if pos is None and pd.notna(last["position"]):
                    pos = str(last["position"])
                price = price if price is not None else int(last["price"])
    except FileNotFoundError:
        pass

    team_name: str | None = None
    team_short: str | None = None
    if team_code is not None and season is not None:
        names, shorts = _team_names(season)
        team_name = names.get(team_code)
        team_short = shorts.get(team_code)

    return PlayerDetail(
        player=identity,
        seasons=seasons,
        season=season,
        team_code=team_code,
        team_name=team_name,
        team_short_name=team_short,
        position=pos,
        price=price,
        history=history_rows,
        upcoming=upcoming,
    )


# --------------------------------------------------------------------------------------
# /my-team/{entry_id} (expensive solver — cached / background, never blocks > 5 s)
# --------------------------------------------------------------------------------------


def _my_team_job_name(entry_id: int) -> str:
    return f"my-team:{entry_id}"


def _my_team_status_url(entry_id: int) -> str:
    return f"/api/my-team/{entry_id}/status"


def _payload_generated_at(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("generated_at")
    if raw is None:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _my_team_response(
    entry_id: int, payload: dict[str, Any], source: str, generated_at: datetime
) -> MyTeamResponse:
    return MyTeamResponse(
        entry_id=entry_id,
        generated_at=generated_at,
        source=source,  # type: ignore[arg-type]
        squad_state=payload.get("squad_state"),
        recommendation=payload["recommendation"],
    )


def _shared_recommendation_payload(entry_id: int) -> tuple[dict[str, Any], datetime] | None:
    """The shared ``recommendation.json`` as a my-team payload, if it matches ``entry_id``.

    A wrapped file declares its entry via an ``entry_id`` key; a bare Recommendation
    dump is assumed to belong to the configured ``FPLAI_ENTRY_ID`` (0 = never matches).
    Freshness is judged by file mtime.
    """
    path = _processed("recommendation.json")
    try:
        data = cache.load_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if "entry_id" in data:
        if int(data["entry_id"]) != entry_id:
            return None
        payload = dict(data)
        payload["recommendation"] = _unwrap(data, "recommendation")
    else:
        if config.settings.entry_id == 0 or config.settings.entry_id != entry_id:
            return None
        payload = {"entry_id": entry_id, "squad_state": None, "recommendation": data}
    return payload, _mtime_utc(path)


def _job_status(job: jobs.Job | None, name: str, entry_id: int, detail: str) -> JobStatusResponse:
    return JobStatusResponse(
        job=name,
        state=(job.state if job is not None else "idle"),  # type: ignore[arg-type]
        started_at=job.started_at if job is not None else None,
        finished_at=job.finished_at if job is not None else None,
        error=job.error if job is not None else None,
        detail=detail,
        status_url=_my_team_status_url(entry_id),
    )


@router.get(
    "/my-team/{entry_id}",
    response_model=MyTeamResponse,
    responses={
        202: {
            "model": JobStatusResponse,
            "description": "Solve running/queued in the background — poll status_url",
        }
    },
)
def my_team(entry_id: int) -> MyTeamResponse | JSONResponse:
    """Squad state + weekly Recommendation for one FPL entry.

    Serves a cached recommendation when one for this ``entry_id`` is younger than 6 h
    (in-memory result, per-entry cache file, or the shared ``recommendation.json``).
    Otherwise kicks a single-flight background solve and answers **202** with a status
    URL — this endpoint never blocks on the solver.
    """
    now = datetime.now(UTC)
    name = _my_team_job_name(entry_id)
    job = jobs.registry.get(name)

    if job is not None and job.state == "running":
        status = _job_status(job, name, entry_id, "solve in progress — poll status_url")
        return JSONResponse(status_code=202, content=status.model_dump(mode="json"))

    def _fresh(ts: datetime | None) -> bool:
        return ts is not None and (now - ts).total_seconds() < RECOMMENDATION_TTL_S

    if job is not None and job.state == "done" and isinstance(job.result, dict):
        generated = _payload_generated_at(job.result)
        if _fresh(generated):
            assert generated is not None
            return _my_team_response(entry_id, job.result, "memory", generated)

    cache_path = jobs.my_team_cache_path(entry_id)
    try:
        payload = cache.load_json(cache_path)
        generated = _payload_generated_at(payload) or _mtime_utc(cache_path)
        if _fresh(generated):
            return _my_team_response(entry_id, payload, "cache", generated)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    shared = _shared_recommendation_payload(entry_id)
    if shared is not None:
        payload, mtime = shared
        if _fresh(mtime):
            return _my_team_response(entry_id, payload, "shared", mtime)

    previous_error = job.error if job is not None and job.state == "error" else None
    job, _started = jobs.registry.start(name, lambda: jobs.solve_my_team(entry_id))
    detail = "solve started — poll status_url, then GET this endpoint again"
    if previous_error is not None:
        detail = f"previous solve failed ({previous_error}); restarted — poll status_url"
    status = _job_status(job, name, entry_id, detail)
    return JSONResponse(status_code=202, content=status.model_dump(mode="json"))


@router.get("/my-team/{entry_id}/status", response_model=JobStatusResponse)
def my_team_status(entry_id: int) -> JobStatusResponse:
    """Status of the background solve for one entry (idle if never started)."""
    name = _my_team_job_name(entry_id)
    job = jobs.registry.get(name)
    detail = {
        None: "no solve started — GET /api/my-team/{entry_id} to kick one",
        "running": "solve in progress",
        "done": "solve finished — GET /api/my-team/{entry_id} for the result",
        "error": "solve failed — GET /api/my-team/{entry_id} to retry",
    }.get(job.state if job is not None else None, "")
    return _job_status(job, name, entry_id, detail)


# --------------------------------------------------------------------------------------
# /refresh (single-flight background pipeline run)
# --------------------------------------------------------------------------------------


@router.post(
    "/refresh",
    status_code=202,
    response_model=RefreshStatusResponse,
    responses={409: {"description": "A refresh is already running"}},
)
def refresh() -> RefreshStatusResponse:
    """Launch the refresh pipeline (snapshot -> pulls -> build -> predict) in the background.

    Single-flight: answers **409** while a refresh is already running.  Progress is
    served by ``GET /api/refresh/status`` from a captured log buffer.
    """
    job, started = jobs.registry.start(jobs.REFRESH_JOB, jobs.run_refresh_captured)
    if not started:
        raise HTTPException(status_code=409, detail="refresh already running")
    return RefreshStatusResponse(
        state="running",  # type: ignore[arg-type]
        started_at=job.started_at,
        finished_at=None,
        error=None,
        log_tail=[],
    )


@router.get("/refresh/status", response_model=RefreshStatusResponse)
def refresh_status(tail: int = Query(default=40, ge=1, le=500)) -> RefreshStatusResponse:
    """Refresh job state plus the last ``tail`` captured log lines."""
    job = jobs.registry.get(jobs.REFRESH_JOB)
    return RefreshStatusResponse(
        state=(job.state if job is not None else "idle"),  # type: ignore[arg-type]
        started_at=job.started_at if job is not None else None,
        finished_at=job.finished_at if job is not None else None,
        error=job.error if job is not None else None,
        log_tail=jobs.refresh_log.tail(tail),
    )


app.include_router(router)
