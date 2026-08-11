"""End-to-end pipeline orchestration: snapshot -> pulls -> build -> models -> optimizer.

Each stage is implemented in its own module; this file only sequences them and
is the single place the CLI and API call into.

Train/predict artifacts live under ``config.MODELS_DIR`` with a
``manifest.json`` describing versions, the training window and headline
metrics.  ``run_predict`` supports two modes:

* **live** (default): predict every upcoming fixture in the next ``horizon``
  gameweeks.  With a live 2026-27 snapshot on disk (``data/live.py``), rows
  for unplayed fixtures are synthesized from the *bootstrap roster* — real
  squads (promoted teams, summer signings), launch prices, reclassified
  positions and availability flags — with the live season's fixtures/teams
  spliced into the processed tables; features come from ALL prior
  player_match history keyed by ``player_code`` (movers keep their form,
  their team/opponent context comes from the new club) and unseen codes get
  the cold-start priors documented in ``fplai.data.live``.  The WC2026
  fatigue list (GWs 1-4), promoted-team strength seeding, new-manager
  uncertainty and a cached the-odds-api snapshot are applied on top.
  Without a live snapshot the legacy roll-forward synthesizer
  (:func:`_future_player_match`) is used.  When no upcoming fixtures exist on
  disk it degrades gracefully to a message and no output.
* **backtest** (``season`` + ``gw`` given): predict historical GWs
  ``gw .. gw+horizon-1`` of ``season`` from strictly-prior information — the
  walk-forward primitive.  Features for each target GW use only matches before
  that GW (enforced by ``features/windows.py``); models should be trained with
  a matching ``before_season``/``before_gw`` cutoff for a leakage-clean eval.

``run_optimize`` consumes whatever ``predictions_gw.parquet`` /
``predictions.parquet`` window is on disk and follows the same duality:

* **live**: the predictions cover upcoming fixtures and the verdict targets
  the next real deadline (with ``--entry-id`` / ``FPLAI_ENTRY_ID``, the plan
  starts from the manager's actual squad via ``optimizer.state.from_entry``).
* **pre-launch demo** (2026-27 not live yet): the predictions are a 2025-26
  backtest window (``fplai predict --season 2025 --gw N`` then
  ``fplai optimize [--season 2025 --gw N]``) and the verdict is an
  initial-squad build over that window — **this is the demo the dashboard
  shows until the 2026-27 game launches**.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    import pandas as pd

    from fplai.data.fpl_api import SeasonState
    from fplai.models.bonus import BonusCalibration
    from fplai.optimizer.plans import Recommendation

console = Console()
logger = logging.getLogger(__name__)

#: Processed tables loaded by :func:`load_processed_tables` (odds is optional).
_PROCESSED_TABLES = ("players", "teams", "fixtures", "player_match", "player_gw")


def _parse_int_spec(spec: str | None, *, lo: int, hi: int, what: str, example: str) -> list[int] | None:
    """Parse ``"30..34,38"``-style int specs into a sorted list (None/blank -> None)."""
    if spec is None or not spec.strip():
        return None
    values: set[int] = set()
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ".." in entry:
            lo_s, _, hi_s = entry.partition("..")
            try:
                start, end = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"bad {what} range {entry!r} (want e.g. {example})") from exc
            if start > end:
                raise ValueError(f"bad {what} range {entry!r}: start > end")
            values.update(range(start, end + 1))
        else:
            try:
                values.add(int(entry))
            except ValueError as exc:
                raise ValueError(f"bad {what} {entry!r} (want e.g. {example})") from exc
    for v in values:
        if not lo <= v <= hi:
            raise ValueError(f"{what} {v} out of supported range {lo}..{hi}")
    return sorted(values)


def parse_seasons(spec: str | None) -> list[int] | None:
    """Parse a CLI season spec into a sorted list of season start years.

    Accepts comma-separated entries where each entry is either a single start
    year (``"2024"``) or an inclusive range (``"2016..2025"``).  Returns
    ``None`` for ``None``/empty input (meaning "use the stage's default").

    >>> parse_seasons("2016..2018,2025")
    [2016, 2017, 2018, 2025]

    Raises:
        ValueError: on malformed entries or years outside 2016..2026.
    """
    return _parse_int_spec(spec, lo=2016, hi=2026, what="season", example="2016..2025")


def parse_gws(spec: str | None) -> list[int] | None:
    """Parse a CLI gameweek spec into a sorted list of GWs (1..38).

    Same grammar as :func:`parse_seasons`: single GWs and/or inclusive ranges,
    comma-separated.  ``None``/blank -> ``None`` ("the stage's default").

    >>> parse_gws("30..32,38")
    [30, 31, 32, 38]

    Raises:
        ValueError: on malformed entries or GWs outside 1..38.
    """
    return _parse_int_spec(spec, lo=1, hi=38, what="GW", example="30..38")


def _table_row_counts(paths: dict[str, Path]) -> dict[str, int]:
    """Row counts of the given parquet tables (from parquet metadata; cheap)."""
    import pyarrow.parquet as pq

    return {name: pq.ParquetFile(path).metadata.num_rows for name, path in sorted(paths.items())}


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def run_snapshot() -> SeasonState:
    """Archive today's FPL API snapshot and report season state (launch detector)."""
    from fplai.data.fpl_api import FplApiClient

    client = FplApiClient()
    snap_dir = client.take_snapshot()
    state = client.season_state()

    console.print(f"[bold]snapshot[/bold]: archived to {snap_dir}")
    console.print(
        f"  season={state.season} ({state.season}-{(state.season + 1) % 100:02d})  "
        f"current_gw={state.current_gw}  next_gw={state.next_gw}  "
        f"players={state.total_players:,}"
    )
    if state.next_deadline_utc is not None:
        console.print(f"  next deadline (UTC): {state.next_deadline_utc:%Y-%m-%d %H:%M}")
    if state.is_live_2026_27:
        console.print(
            "[bold red]  2026-27 IS LIVE[/bold red] — re-verify scoring/rules assumptions "
            "against the new bootstrap (see FPL_KNOWLEDGE UNCERTAIN items)"
        )
    else:
        console.print(f"  2026-27 launch check: not live yet (API still serving {state.season})")
    return state


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------


def run_backfill(seasons: Sequence[int] | None = None) -> None:
    """Download historical raw data: vaastav, football-data odds, Understat, ClubElo.

    Args:
        seasons: Season start years to fetch (default: every vaastav season,
            2016..2025).  Idempotent — already-downloaded files are skipped.
    """
    from fplai.data import clubelo, football_data, understat, vaastav

    chosen: list[int] = sorted(seasons) if seasons is not None else list(vaastav.SEASONS)

    console.print(f"[bold]backfill[/bold]: seasons {', '.join(str(s) for s in chosen)}")

    console.print("  vaastav historical CSVs ...")
    vaastav.download_all(chosen)

    console.print("  football-data.co.uk E0.csv odds ...")
    for season in chosen:
        football_data.download_e0(season)

    console.print("  Understat league JSON ...")
    us_client = understat.UnderstatClient()
    for season in chosen:
        if season <= understat.HISTORY_END_YEAR:
            us_client.fetch_league_season(season)
        else:
            console.print(f"    [yellow]skipping Understat {season} (not yet served)[/yellow]")

    console.print("  ClubElo snapshot + current PL club histories ...")
    elo = clubelo.EloClient()
    today = dt.date.today()
    snap = elo.pl_snapshot(today)
    for club in snap["Club"]:
        elo.team_history(str(club))
    console.print(f"    {len(snap)} PL clubs archived under raw/clubelo/")

    console.print("[bold]backfill[/bold]: done")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def run_build(seasons: Sequence[int] | None = None) -> dict[str, Path]:
    """Build every processed parquet table from raw data already on disk.

    Builds teams/players/fixtures/player_match/player_gw over the requested
    seasons (default: every season with a raw vaastav directory), plus
    odds.parquet over whichever of those seasons have a raw E0.csv.  Never
    touches the network.  Returns table name -> written path.
    """
    from fplai.data import build, crosswalk, football_data

    resolved = crosswalk.resolve_seasons(seasons)
    console.print(
        f"[bold]build[/bold]: seasons {', '.join(str(s) for s in resolved)} "
        "(from raw vaastav dirs)"
    )
    paths = build.build_all(resolved)

    odds_seasons = [s for s in resolved if football_data.e0_path(s).exists()]
    if odds_seasons:
        paths["odds"] = football_data.build_odds_table(odds_seasons)
    else:
        console.print(
            "  [yellow]no raw football-data E0.csv files for these seasons — "
            "skipping odds.parquet (run `fplai backfill` first)[/yellow]"
        )

    for name, count in _table_row_counts(paths).items():
        console.print(f"  {name}.parquet: {count:,} rows")
    return paths


# ---------------------------------------------------------------------------
# ingest (played live-season GWs -> player_match/player_gw splice)
# ---------------------------------------------------------------------------


def run_ingest(processed_dir: Path | None = None) -> Any | None:
    """Splice played live-season GW outcomes into player_match/player_gw.

    Sources ``element-summary`` history rows via the FPL API (see
    ``fplai.data.ingest``).  No-op (and no network) when nothing has been
    played or every played GW is already frozen on disk.  Must run AFTER
    ``build`` (which drops live-season rows) and BEFORE ``predict``.
    """
    from fplai.data.fpl_api import FplApiClient
    from fplai.data.ingest import ingest_played

    ctx = _live_context()
    if ctx is None:
        console.print("[bold]ingest[/bold]: no live snapshot — nothing to do")
        return None
    report = ingest_played(
        FplApiClient(),
        ctx,
        processed_dir=processed_dir,
        on_progress=lambda done, total: (
            console.print(f"  element summaries: {done}/{total}") if done % 100 == 0 else None
        ),
    )
    if report is None:
        console.print("[bold]ingest[/bold]: no unfrozen played GWs — up to date")
        return None
    console.print(
        f"[bold]ingest[/bold]: season {report.season} GWs {report.gws} — "
        f"{report.n_rows:,} player-fixture rows from {report.n_players} players "
        f"({report.n_swept} summaries swept); frozen: {report.frozen_gws or 'none yet'}"
    )
    if report.unmatched_elements:
        console.print(
            f"  [yellow]{len(report.unmatched_elements)} swept elements missing from the "
            f"roster (ids {report.unmatched_elements[:5]}…)[/yellow]"
        )
    return report


# ---------------------------------------------------------------------------
# refresh (snapshot -> incremental pulls -> build -> model stages)
# ---------------------------------------------------------------------------


def _refresh_pulls(state: SeasonState) -> None:
    """Incremental data pulls for the current season (best-effort per source).

    The FPL snapshot and the build are hard requirements of ``refresh``; the
    auxiliary sources (vaastav mirror, Understat, football-data, ClubElo) are
    refreshed best-effort with a visible warning on failure so a single
    upstream outage doesn't take down the daily refresh.
    """
    from fplai.data import clubelo, football_data, understat, vaastav

    season = state.season
    in_progress = state.next_gw is not None

    if season in vaastav.SEASONS:
        try:
            console.print(
                f"  vaastav {season} ({'in-season force refresh' if in_progress else 'fill-if-missing'}) ..."
            )
            vaastav.download_season(season, force=in_progress)
        except Exception as exc:  # noqa: BLE001 - best-effort aux pull, warned loudly
            console.print(f"  [yellow]vaastav refresh failed: {exc}[/yellow]")
    else:
        console.print(f"  [yellow]vaastav has no {season} mirror yet — skipping[/yellow]")

    try:
        console.print(f"  football-data E0 {season} ...")
        football_data.download_e0(season, force=in_progress)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]football-data refresh failed: {exc}[/yellow]")

    try:
        console.print(f"  Understat league {season} ...")
        understat.UnderstatClient().fetch_league_season(season, force=in_progress)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]Understat refresh failed: {exc}[/yellow]")

    try:
        console.print("  ClubElo snapshot ...")
        clubelo.EloClient().pl_snapshot(dt.date.today())
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]ClubElo refresh failed: {exc}[/yellow]")


def _print_launch_watch(state: SeasonState) -> None:
    """Prominent end-of-refresh season status: the 2026-27 launch watch."""
    from rich.panel import Panel

    from fplai import rules

    if state.is_live_2026_27:
        deadline = (
            f"\nNext deadline (UTC): {state.next_deadline_utc:%Y-%m-%d %H:%M} (GW{state.next_gw})"
            if state.next_deadline_utc is not None
            else ""
        )
        console.print(
            Panel(
                f"[bold green]2026-27 IS LIVE[/bold green] — predictions and "
                f"recommendations now target the real season.{deadline}\n"
                "Re-verify FPL_KNOWLEDGE UNCERTAIN items and retrain with `fplai train`.",
                title="launch watch",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]2026-27 NOT LIVE yet[/bold yellow] — the FPL API still serves "
                f"{rules.season_label(state.season)}.\n"
                "Running in pre-launch demo mode: predictions/recommendations come from the "
                "2025-26 backtest window (the demo the dashboard shows until launch).\n"
                "GW1 deadline expected 2026-08-21 17:30 UTC — keep `fplai refresh` running daily.",
                title="launch watch",
            )
        )


def run_refresh() -> None:
    """Pull fresh data from all sources, rebuild tables, then the model stages.

    The data portion (snapshot -> incremental pulls -> build) is fully wired
    and must succeed; the model stages are best-effort so `fplai refresh`
    stays exit-0: train is never run automatically (run `fplai train`);
    predict, optimize and then the chip-timing simulation run when artifacts
    exist, each logging and continuing on failure.  In live mode predict
    defaults its window to the end of the current chip window (GW19 pre-GW20)
    so the simulation has full coverage.  In pre-launch mode the refresh ends
    by printing the launch-watch status (season state) prominently.
    """
    state = run_snapshot()
    console.print("[bold]refresh[/bold]: incremental pulls")
    _refresh_pulls(state)
    run_build()
    from fplai import config

    if state.is_live_2026_27:
        # AFTER build (which rebuilds player_match from raw and drops live-season
        # rows), BEFORE predict (whose form features need the played GWs).
        try:
            run_ingest()
        except Exception as exc:  # noqa: BLE001 - refresh must stay exit-0
            console.print(f"[bold]ingest[/bold]: [yellow]failed: {exc}[/yellow] (skipping)")

    if (config.MODELS_DIR / "manifest.json").exists():
        console.print("[bold]train[/bold]: using existing artifacts (retrain with `fplai train`)")
        through: int | None = None
        if state.is_live_2026_27 and state.next_gw is not None:
            from fplai.models.sampler import chip_window_end

            through = chip_window_end(state.season, int(state.next_gw))
        try:
            run_predict(through_gw=through)
        except Exception as exc:  # noqa: BLE001 - refresh must stay exit-0
            console.print(f"[bold]predict[/bold]: [yellow]failed: {exc}[/yellow] (skipping)")
        try:
            run_optimize()
        except Exception as exc:  # noqa: BLE001 - refresh must stay exit-0
            console.print(f"[bold]optimize[/bold]: [yellow]failed: {exc}[/yellow] (skipping)")
        try:
            run_simulate()
        except Exception as exc:  # noqa: BLE001 - refresh must stay exit-0
            console.print(f"[bold]simulate[/bold]: [yellow]failed: {exc}[/yellow] (skipping)")
    else:
        console.print("[bold]train[/bold]: no model artifacts yet — run `fplai train` (skipping)")
        console.print("[bold]predict[/bold]: skipped (no model artifacts)")
        console.print("[bold]optimize[/bold]: skipped (no model artifacts)")
        console.print("[bold]simulate[/bold]: skipped (no model artifacts)")
    _print_launch_watch(state)
    console.print("[bold]refresh[/bold]: data refresh complete")


# ---------------------------------------------------------------------------
# model stages: shared helpers
# ---------------------------------------------------------------------------


def load_processed_tables(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load every canonical processed parquet table into a dict.

    ``odds`` is optional (missing file -> absent key); everything else raises
    with a pointer to ``fplai build`` when absent.
    """
    import pandas as pd

    from fplai import config

    processed = processed_dir if processed_dir is not None else config.PROCESSED_DIR
    tables: dict[str, pd.DataFrame] = {}
    for name in _PROCESSED_TABLES:
        path = processed / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing — run `fplai build` first")
        tables[name] = pd.read_parquet(path)
    odds_path = processed / "odds.parquet"
    if odds_path.exists():
        tables["odds"] = pd.read_parquet(odds_path)
    return tables


def _cutoff_mask(
    df: pd.DataFrame, before_season: int | None, before_gw: int | None
) -> pd.Series:
    """Boolean mask of rows strictly before the ``(before_season, before_gw)`` cutoff.

    ``before_gw=None`` means "before the whole season".  No cutoff -> all True.
    """
    if before_season is None:
        return df["season"] == df["season"]  # all-True, index-aligned
    if before_gw is None:
        return df["season"] < before_season
    return (df["season"] < before_season) | (
        (df["season"] == before_season) & (df["gw"] < before_gw)
    )


def _team_train_end(
    fixtures: pd.DataFrame, before_season: int | None, before_gw: int | None
) -> pd.Timestamp | None:
    """First kickoff at/after the cutoff — the TeamModel ``train_end`` guard."""
    excluded = fixtures.loc[~_cutoff_mask(fixtures, before_season, before_gw), "kickoff_utc"]
    if excluded.empty:
        return None
    import pandas as pd

    return pd.Timestamp(excluded.min())


def _bonus_calibration_season(pm_train: pd.DataFrame, min_rows: int = 2000) -> int | None:
    """Latest train season with enough played rows to fit the bonus calibration."""
    played = pm_train[pm_train["minutes"] > 0]
    counts = played.groupby("season").size()
    eligible = counts[counts >= min_rows]
    return int(eligible.index.max()) if len(eligible) else None


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def run_train(
    seasons: Sequence[int] | None = None,
    *,
    before_season: int | None = None,
    before_gw: int | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train every model component on the processed tables and save artifacts.

    Args:
        seasons: Restrict training to these season start years (default: all
            seasons on disk).
        before_season/before_gw: Walk-forward cutoff — train strictly before
            GW ``before_gw`` of ``before_season`` (``before_gw=None`` = before
            the whole season).  The TeamModel additionally gets a
            ``train_end`` timestamp at the first excluded kickoff.
        models_dir: Artifact root (default ``config.MODELS_DIR``).  Components
            are saved under ``<models_dir>/{minutes,team,rates,bonus}`` plus a
            ``manifest.json``.

    Returns:
        The manifest dict (also written to ``manifest.json``).
    """
    from fplai import config
    from fplai.features.windows import build_feature_frame
    from fplai.models import LgbMinutesModel, RatesModel, TeamModel, bonus

    out_dir = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_processed_tables()
    if seasons is not None:
        wanted = set(seasons)
        for name in ("player_match", "player_gw", "fixtures", "teams"):
            tables[name] = tables[name][tables[name]["season"].isin(wanted)].copy()
        if "odds" in tables:
            tables["odds"] = tables["odds"][tables["odds"]["season"].isin(wanted)].copy()

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    features = build_feature_frame(tables, target="match")
    timings["features"] = round(time.perf_counter() - t0, 2)
    train_rows = features[_cutoff_mask(features, before_season, before_gw)]
    pm_train = tables["player_match"][
        _cutoff_mask(tables["player_match"], before_season, before_gw)
    ]
    train_seasons = sorted(int(s) for s in train_rows["season"].unique())
    console.print(
        f"[bold]train[/bold]: {len(train_rows):,} feature rows over seasons "
        f"{train_seasons[0]}..{train_seasons[-1]}"
        + (
            f" (cutoff: before {before_season} GW{before_gw})"
            if before_season is not None and before_gw is not None
            else (f" (cutoff: before season {before_season})" if before_season else "")
        )
    )

    # -- minutes (v1 two-stage LightGBM) ---------------------------------
    t0 = time.perf_counter()
    minutes_model = LgbMinutesModel().fit(train_rows)
    timings["minutes_fit"] = round(time.perf_counter() - t0, 2)
    last_season_rows = train_rows[train_rows["season"] == train_seasons[-1]]
    minutes_eval = minutes_model.evaluate(last_season_rows)
    minutes_metrics = {
        "scope": f"in-sample last train season ({train_seasons[-1]})",
        "n": minutes_eval["n"],
        "bucket_log_loss": round(minutes_eval["bucket_log_loss"], 4),
        "brier_p0": round(minutes_eval["brier_p0"], 4),
    }
    minutes_model.save(out_dir / "minutes")
    console.print(
        f"  minutes: LgbMinutesModel fit in {timings['minutes_fit']}s — "
        f"log-loss {minutes_metrics['bucket_log_loss']} on {train_seasons[-1]} (in-sample)"
    )

    # -- team (Dixon-Coles) ----------------------------------------------
    t0 = time.perf_counter()
    team_model = TeamModel().fit(
        tables["fixtures"],
        tables.get("odds"),
        train_end=_team_train_end(tables["fixtures"], before_season, before_gw),
        teams=tables["teams"],
    )
    timings["team_fit"] = round(time.perf_counter() - t0, 2)
    team_metrics = {
        "n_matches": team_model.n_matches_,
        "gamma": round(math.exp(team_model.log_gamma_), 4),
        "rho": round(team_model.rho_, 4),
    }
    team_model.save(out_dir / "team")
    console.print(
        f"  team: Dixon-Coles fit on {team_model.n_matches_:,} matches "
        f"in {timings['team_fit']}s"
    )

    # -- rates (per-90 LightGBM Poisson) ---------------------------------
    t0 = time.perf_counter()
    rates_model = RatesModel().fit(train_rows)
    timings["rates_fit"] = round(time.perf_counter() - t0, 2)
    rates_model.save(out_dir / "rates")
    console.print(f"  rates: RatesModel fit in {timings['rates_fit']}s")

    # -- bonus calibration ------------------------------------------------
    t0 = time.perf_counter()
    cal_season = _bonus_calibration_season(pm_train)
    if cal_season is not None:
        calibration = bonus.calibrate(pm_train, cal_season)
    else:
        calibration = bonus.DEFAULT_CALIBRATION
    timings["bonus_calibrate"] = round(time.perf_counter() - t0, 2)
    bonus_dir = out_dir / "bonus"
    bonus_dir.mkdir(parents=True, exist_ok=True)
    (bonus_dir / "calibration.json").write_text(
        json.dumps(
            {
                "season": cal_season,
                "bias": calibration.bias,
                "sigma_intercept": calibration.sigma_intercept,
                "sigma_slope": calibration.sigma_slope,
                "sigma_floor": calibration.sigma_floor,
            },
            indent=1,
        )
    )
    console.print(
        f"  bonus: calibration fitted on season {cal_season}"
        if cal_season is not None
        else "  bonus: too little data — using DEFAULT_CALIBRATION"
    )

    manifest: dict[str, Any] = {
        "schema": 1,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "train_window": {
            "seasons": train_seasons,
            "before_season": before_season,
            "before_gw": before_gw,
            "n_feature_rows": int(len(train_rows)),
            "n_team_matches": int(team_model.n_matches_),
        },
        "components": {
            "minutes": {"class": "LgbMinutesModel", "version": "v1", "metrics": minutes_metrics},
            "team": {"class": "TeamModel", "version": "v1", "metrics": team_metrics},
            "rates": {"class": "RatesModel", "version": "v1"},
            "bonus": {"calibration_season": cal_season},
        },
        "timings_s": timings,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    console.print(f"[bold]train[/bold]: artifacts saved to {out_dir}")
    return manifest


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _load_artifacts(
    models_dir: Path | None = None,
) -> tuple[dict[str, Any], Any, Any, Any, BonusCalibration]:
    """Load manifest + the four model artifacts; raise with guidance if absent."""
    from fplai import config
    from fplai.models import LgbMinutesModel, RatesModel, TeamModel
    from fplai.models.bonus import DEFAULT_CALIBRATION, BonusCalibration

    root = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is missing — run `fplai train` first")
    manifest = json.loads(manifest_path.read_text())
    minutes_model = LgbMinutesModel.load(root / "minutes")
    team_model = TeamModel.load(root / "team")
    rates_model = RatesModel.load(root / "rates")
    cal_path = root / "bonus" / "calibration.json"
    if cal_path.exists():
        raw = json.loads(cal_path.read_text())
        calibration = BonusCalibration(
            bias=raw["bias"],
            sigma_intercept=raw["sigma_intercept"],
            sigma_slope=raw["sigma_slope"],
            sigma_floor=raw["sigma_floor"],
        )
    else:
        calibration = DEFAULT_CALIBRATION
    return manifest, minutes_model, team_model, rates_model, calibration


def _live_context() -> Any | None:
    """Best-effort :class:`~fplai.data.live.LiveContext` from the newest snapshot.

    ``None`` when no live (2026-27+) snapshot exists, or when the snapshot is
    unusable — live predict then falls back to the roll-forward synthesizer
    with a loud warning rather than taking down the daily refresh.
    """
    from fplai.data import live

    try:
        return live.load_live_context()
    except Exception as exc:  # noqa: BLE001 - degrade to roll-forward, warned
        console.print(
            f"  [yellow]live snapshot unusable ({exc}) — "
            "falling back to roll-forward rosters[/yellow]"
        )
        return None


def _future_player_match(pm: pd.DataFrame, target_fx: pd.DataFrame) -> pd.DataFrame:
    """Synthesize player_match rows for unplayed fixtures (roll-forward fallback).

    Rosters roll forward from each player's most recent appearance: a player
    belongs to the squad of the team of their latest ``player_match`` row,
    keeping that row's position/price/element id.  Only players seen in the
    target season or the season before are considered (no resurrecting
    long-departed players).  All outcome columns are NaN — the feature builder
    skips NaNs, so these rows contribute nothing to anyone's history.

    From the 2026-27 relaunch onward the bootstrap-roster synthesizer
    (:func:`fplai.data.live.bootstrap_player_match`) replaces this for the
    live season — this path remains for pre-relaunch data and as the fallback
    when no live snapshot is available.
    """
    import numpy as np
    import pandas as pd

    from fplai import rules

    latest = (
        pm.sort_values(["player_code", "season", "gw"], kind="stable")
        .groupby("player_code", as_index=False)
        .tail(1)
    )
    min_season = int(target_fx["season"].min()) - 1
    latest = latest[latest["season"] >= min_season]

    rows: list[pd.DataFrame] = []
    for fx_row in target_fx.itertuples():
        for team_code, opponent_code, was_home in (
            (fx_row.home_team_code, fx_row.away_team_code, True),
            (fx_row.away_team_code, fx_row.home_team_code, False),
        ):
            roster = latest[latest["team_code"] == team_code]
            if roster.empty:
                continue
            flags = rules.SEASON_FLAGS.get(int(fx_row.season))
            synth = pd.DataFrame(
                {
                    "season": int(fx_row.season),
                    "gw": int(fx_row.gw),
                    "fpl_fixture_id": int(fx_row.fpl_fixture_id),
                    "player_code": roster["player_code"].to_numpy(),
                    "fpl_element_id": roster["fpl_element_id"].to_numpy(),
                    "team_code": int(team_code),
                    "opponent_code": int(opponent_code),
                    "was_home": was_home,
                    "position": roster["position"].to_numpy(),
                    "price": roster["price"].to_numpy(),
                    "empty_stadium": False,
                    "void_gw": False,
                    "subs_regime": flags.subs_regime if flags is not None else 5,
                    "stint_id": roster["stint_id"].to_numpy(),
                }
            )
            rows.append(synth)
    if not rows:
        return pm.iloc[0:0].copy()
    out = pd.concat(rows, ignore_index=True)
    for col in pm.columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[pm.columns]


def _predict_frames(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    models: tuple[Any, Any, Any, BonusCalibration],
    use_odds: bool,
    live_adjust: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run minutes/team/rates + assemble over one already-selected feature frame.

    Returns ``(pred, gw_pred, team_pred)`` — the third frame is the TeamModel's
    per-fixture prediction (goal lambdas, clean-sheet and 1X2 probabilities,
    odds-blend flag) joined with fixture context; ``run_predict`` persists it as
    ``team_fixtures.parquet`` for the fixture-outlook surfaces.

    ``live_adjust`` (a :class:`fplai.data.live.LiveAdjustments`) switches the
    minutes/rates paths to the live policy: WC2026 fatigue dampening on GWs
    1-4, cold-start priors for unseen player codes and availability gating —
    see the ``fplai.data.live`` module docstring for the full policy.
    """
    import numpy as np
    import pandas as pd

    from fplai.models import aggregate_gw, assemble_xp
    from fplai.models.team import attach_fixture_ids

    minutes_model, team_model, rates_model, calibration = models

    target_keys = features[["season", "gw"]].drop_duplicates()
    fx_target = fixtures.merge(target_keys, on=["season", "gw"])
    fx_target = fx_target[~fx_target["void"].astype(bool)]

    team_pred = team_model.predict_fixtures(fx_target)
    if use_odds and "odds" in tables and len(tables["odds"]):
        try:
            odds = attach_fixture_ids(tables["odds"], fixtures, tables["teams"])
            team_pred = team_model.blend_odds(team_pred, odds)
            n_blended = int(team_pred["odds_blended"].sum())
            if n_blended:
                console.print(f"  odds blend: {n_blended} fixtures matched market prices")
        except Exception as exc:  # noqa: BLE001 - odds are optional enrichment
            console.print(f"  [yellow]odds blend skipped: {exc}[/yellow]")

    if live_adjust is not None:
        from fplai.data import live

        minutes_pred = live.live_minutes_predict(minutes_model, features, live_adjust)
        rates_pred = live.apply_cold_start_rates(
            rates_model.predict(features), features, live_adjust
        )
    else:
        minutes_pred = minutes_model.predict(features)
        rates_pred = rates_model.predict(features)

    xp_parts: list[pd.DataFrame] = []
    for season, feats_s in features.groupby("season"):
        xp_parts.append(
            assemble_xp(
                minutes_pred[minutes_pred["season"] == season],
                team_pred[team_pred["season"] == season],
                rates_pred[rates_pred["season"] == season],
                feats_s,
                int(season),
                bonus_calibration=calibration,
            )
        )
    xp = pd.concat(xp_parts, ignore_index=True)

    # Per-fixture output: xp decomposition + context + minutes distribution.
    extras = features[
        ["season", "gw", "player_code", "fpl_fixture_id", "team_code", "opponent_code",
         "was_home", "position", "price"]
    ]
    pred = xp.merge(extras, on=["season", "gw", "player_code", "fpl_fixture_id"], how="left")
    pred = pred.merge(
        minutes_pred[["season", "gw", "player_code", "fpl_fixture_id", "q0", "q1", "q2",
                      "mu1", "mu2"]],
        on=["season", "gw", "player_code", "fpl_fixture_id"],
        how="left",
    )

    # Per-GW output: summed components + P(0 minutes across the whole GW).
    gw_pred = aggregate_gw(xp)
    q0_gw = (
        pred.assign(_logq0=np.log(pred["q0"].clip(1e-12)))
        .groupby(["season", "gw", "player_code"], as_index=False)["_logq0"]
        .sum()
    )
    q0_gw["q0"] = np.exp(q0_gw.pop("_logq0"))
    gw_pred = gw_pred.merge(q0_gw, on=["season", "gw", "player_code"], how="left")
    last_ctx = (
        pred.sort_values(["season", "gw", "player_code", "fpl_fixture_id"])
        .groupby(["season", "gw", "player_code"], as_index=False)
        .tail(1)[["season", "gw", "player_code", "team_code", "position", "price"]]
    )
    gw_pred = gw_pred.merge(last_ctx, on=["season", "gw", "player_code"], how="left")
    players = tables["players"][["player_code", "web_name"]].drop_duplicates("player_code")
    gw_pred = gw_pred.merge(players, on="player_code", how="left")

    # Team-level per-fixture outlook: model probabilities + fixture context.
    ctx_cols = ["season", "gw", "fpl_fixture_id", "kickoff_utc",
                "home_team_code", "away_team_code"]
    team_out = fx_target[ctx_cols].merge(
        team_pred, on=["season", "fpl_fixture_id"], how="left"
    )
    if "odds_blended" not in team_out.columns:
        team_out["odds_blended"] = False
    return pred, gw_pred, team_out


def _captaincy_frame(
    pred: pd.DataFrame,
    gw_pred: pd.DataFrame,
    *,
    n_draws: int = 2000,
    seed: int = 0,
    top_k: int = 8,
) -> pd.DataFrame | None:
    """Distributional captaincy comparison for the window's first GW.

    Samples joint realized points (shared fixture scorelines preserved — two
    candidates in the same match are correlated) for the ``top_k`` next-GW xP
    candidates and summarizes what a point estimate hides: haul/blank tails,
    P(best captain of the set) with ties split evenly, and P(outscores the
    top-xP pick).  Deterministic under ``seed``.  ``None`` when the window has
    fewer than two viable candidates.
    """
    import numpy as np
    import pandas as pd

    from fplai.models.sampler import PointsSampler

    season = int(gw_pred["season"].max())
    sg = gw_pred[gw_pred["season"] == season]
    first_gw = int(sg["gw"].min())
    pool = sg[(sg["gw"] == first_gw) & (sg["q0"].fillna(1.0) <= 0.5)]
    cands = pool.nlargest(top_k, "xp")[["player_code", "xp"]]
    if len(cands) < 2:
        return None
    codes = [int(c) for c in cands["player_code"]]
    rows = pred[
        (pred["season"] == season)
        & (pred["gw"] == first_gw)
        & (pred["player_code"].isin(codes))
    ]
    if rows.empty:
        return None
    sample = PointsSampler().sample_gw(rows, n_draws, seed)
    order = {int(c): i for i, c in enumerate(sample.index["player_code"])}
    pts = sample.points[:, [order[c] for c in codes]].astype(np.int32)  # [n, k]

    best = pts.max(axis=1, keepdims=True)
    is_best = pts == best
    p_best = (is_best / is_best.sum(axis=1, keepdims=True)).mean(axis=0)
    top_pts = pts[:, 0]  # cands sorted by xp desc — column 0 is the top-xP pick
    beats_top = (pts > top_pts[:, None]).mean(axis=0) + 0.5 * (
        pts == top_pts[:, None]
    ).mean(axis=0)
    beats_top[0] = np.nan

    return pd.DataFrame(
        {
            "season": season,
            "gw": first_gw,
            "player_code": codes,
            "xp": cands["xp"].to_numpy(dtype=float),
            "mean_pts": pts.mean(axis=0),
            "sd_pts": pts.std(axis=0),
            "p_haul": (pts >= 10).mean(axis=0),
            "p_blank": (pts <= 2).mean(axis=0),
            "p_best": p_best,
            "p_beats_top": beats_top,
            "n_draws": n_draws,
        }
    )


def run_predict(
    season: int | None = None,
    gw: int | None = None,
    *,
    horizon: int | None = None,
    through_gw: int | None = None,
    use_odds: bool = True,
    models_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path] | None:
    """Generate xP predictions and write predictions(.gw).parquet.

    Live mode (no ``season``/``gw``): predict all upcoming fixtures within the
    next ``horizon`` GWs; returns ``None`` (with a message) when the fixture
    list holds nothing upcoming (e.g. between seasons — run a backtest GW
    instead).  Backtest mode (``season`` + ``gw``): predict historical GWs
    ``gw .. gw+horizon-1`` of ``season`` from strictly-prior information.

    ``through_gw`` (live mode only; mutually exclusive with ``horizon``)
    extends the window through that GW inclusive — the season-simulation
    stage wants predictions to the end of the current chip window
    (``fplai predict --through-gw 19``); ``run_refresh`` passes it
    automatically in live mode.

    Returns ``{"predictions": path, "predictions_gw": path}`` or ``None``.
    """
    import pandas as pd

    from fplai import config
    from fplai.features.windows import build_feature_frame

    if (season is None) != (gw is None):
        raise ValueError("season and gw must be given together (backtest mode) or not at all")
    if through_gw is not None:
        if season is not None:
            raise ValueError("through_gw is a live-mode option; use horizon in backtest mode")
        if horizon is not None:
            raise ValueError("give either horizon or through_gw, not both")
    horizon = horizon if horizon is not None else config.settings.horizon_gws
    out = Path(out_dir) if out_dir is not None else config.PROCESSED_DIR

    tables = load_processed_tables()
    fixtures = tables["fixtures"]
    ctx: Any | None = None

    if season is not None and gw is not None:
        mode = f"backtest {season} GW{gw}+ (horizon {horizon})"
        features = build_feature_frame(tables, target="match")
        gws = sorted(
            g for g in features.loc[features["season"] == season, "gw"].unique() if g >= gw
        )[:horizon]
        if not gws:
            console.print(f"[bold]predict[/bold]: no rows for season {season} GW>={gw}")
            return None
        rows = features[(features["season"] == season) & (features["gw"].isin(gws))]
    else:
        ctx = _live_context()
        if ctx is not None:
            from fplai.data import live

            live_odds = None
            if use_odds:
                try:
                    live_odds = live.fetch_live_odds(ctx)
                except Exception as exc:  # noqa: BLE001 - odds are optional enrichment
                    console.print(f"  [yellow]live odds unavailable: {exc}[/yellow]")
            tables = live.augment_tables(tables, ctx, odds=live_odds)
            fixtures = tables["fixtures"]
            console.print(
                f"  live snapshot {ctx.snap_dir.name}: season {ctx.season}, "
                f"{len(ctx.roster)} rostered players, "
                f"{len(live_odds) if live_odds is not None else 0} odds rows"
            )
        now = pd.Timestamp.now(tz="UTC")
        upcoming = fixtures[
            ~fixtures["finished"].astype(bool)
            & ~fixtures["void"].astype(bool)
            & (fixtures["kickoff_utc"] >= now - pd.Timedelta("12h"))
        ]
        if upcoming.empty:
            console.print(
                "[bold]predict[/bold]: no upcoming fixtures on disk — nothing to predict. "
                "Use backtest mode, e.g. `fplai predict --season 2025 --gw 30`."
            )
            return None
        gw_keys = (
            upcoming.groupby(["season", "gw"], as_index=False)["kickoff_utc"]
            .min()
            .sort_values("kickoff_utc")
        )
        if through_gw is not None:
            from fplai.models.sampler import horizon_through_gw

            horizon = horizon_through_gw(int(gw_keys.iloc[0]["gw"]), through_gw)
        next_keys = gw_keys.head(horizon)[["season", "gw"]]
        target_fx = upcoming.merge(next_keys, on=["season", "gw"])
        mode = (
            f"live: {len(target_fx)} upcoming fixtures over GWs "
            f"{sorted(next_keys['gw'].tolist())}"
        )
        pm = tables["player_match"]
        if ctx is not None and (target_fx["season"] == ctx.season).any():
            # Bootstrap-grounded rosters for the live season; roll-forward only
            # for any residual pre-relaunch fixtures in the window.
            live_fx = target_fx[target_fx["season"] == ctx.season]
            rest_fx = target_fx[target_fx["season"] != ctx.season]
            parts = [live.bootstrap_player_match(ctx, live_fx, pm)]
            if len(rest_fx):
                parts.append(_future_player_match(pm, rest_fx))
            synth = pd.concat(parts, ignore_index=True)
        else:
            synth = _future_player_match(pm, target_fx)
        # Guard against duplicating rows the build already produced (keyed per
        # fixture so a played DGW leg never suppresses the upcoming one).
        played_keys = set(
            zip(pm["season"], pm["gw"], pm["player_code"], pm["fpl_fixture_id"], strict=True)
        )
        synth = synth[
            [
                (s, g, p, f) not in played_keys
                for s, g, p, f in zip(
                    synth["season"], synth["gw"], synth["player_code"],
                    synth["fpl_fixture_id"], strict=True,
                )
            ]
        ]
        aug = dict(tables)
        aug["player_match"] = pd.concat([pm, synth], ignore_index=True)
        if ctx is not None:
            live_gws = sorted(
                int(g)
                for g in next_keys.loc[next_keys["season"] == ctx.season, "gw"].unique()
            )
            if live_gws:
                aug["availability"] = live.availability_frame(ctx, live_gws)
        features = build_feature_frame(aug, target="match")
        want = set(
            zip(synth["season"], synth["gw"], synth["player_code"], synth["fpl_fixture_id"],
                strict=True)
        )
        rows = features[
            [
                (s, g, p, f) in want
                for s, g, p, f in zip(
                    features["season"], features["gw"], features["player_code"],
                    features["fpl_fixture_id"], strict=True,
                )
            ]
        ]

    console.print(f"[bold]predict[/bold]: {mode} — {len(rows):,} player-fixture rows")
    out.mkdir(parents=True, exist_ok=True)
    _manifest, minutes_model, team_model, rates_model, calibration = _load_artifacts(models_dir)
    live_adjust = None
    if ctx is not None:
        from fplai.data import live

        seed_report = live.seed_team_model(team_model, ctx)
        live_adjust = live.build_adjustments(ctx, tables["player_match"])
        live.persist_live_tables(ctx, tables)
        console.print(
            f"  live adjustments: {len(seed_report['promoted'])} promoted clubs seeded, "
            f"{len(seed_report['new_manager'])} new-manager clubs widened, "
            f"{len(live_adjust.cold_codes)} cold-start players, "
            f"fatigue dampening for {len(live_adjust.fatigue)} WC2026 players "
            f"(GWs {live_adjust.fatigue_gws[0]}-{live_adjust.fatigue_gws[-1]})"
        )
    pred, gw_pred, team_fx = _predict_frames(
        rows, fixtures, tables, (minutes_model, team_model, rates_model, calibration), use_odds,
        live_adjust=live_adjust,
    )
    paths = {
        "predictions": out / "predictions.parquet",
        "predictions_gw": out / "predictions_gw.parquet",
        "team_fixtures": out / "team_fixtures.parquet",
    }
    pred.to_parquet(paths["predictions"], index=False)
    gw_pred.to_parquet(paths["predictions_gw"], index=False)
    team_fx.to_parquet(paths["team_fixtures"], index=False)
    try:
        cap = _captaincy_frame(pred, gw_pred)
    except Exception as exc:  # noqa: BLE001 - optional artifact must not fail predict
        console.print(f"  [yellow]captaincy sample skipped: {exc}[/yellow]")
        cap = None
    if cap is not None:
        paths["captaincy"] = out / "captaincy.parquet"
        cap.to_parquet(paths["captaincy"], index=False)

    first = gw_pred[gw_pred["gw"] == gw_pred["gw"].min()]
    top = first.nlargest(10, "xp")[["web_name", "position", "price", "xp"]]
    console.print(
        f"  top-10 xP for GW{int(gw_pred['gw'].min())}: "
        + ", ".join(
            f"{r.web_name} ({r.position}, {r.price / 10:.1f}m, {r.xp:.2f})"
            for r in top.itertuples()
        )
    )
    console.print(
        f"[bold]predict[/bold]: wrote {paths['predictions']} ({len(pred):,} rows) and "
        f"{paths['predictions_gw']} ({len(gw_pred):,} rows)"
    )
    return paths


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


def _optimizer_prices(
    xp_window: pd.DataFrame,
    processed: Path,
    season: int,
    start_gw: int,
    pred_fixture: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the optimizer's ``prices`` frame: one row per player in the xp window.

    Columns: ``player_code, price, position, team_code, web_name`` (price in
    £0.1m units) per the ``optimizer/milp.py`` contract.  In live mode
    (``live_roster.parquet`` present for the target season, written by
    ``run_predict``) the bootstrap roster is the primary source — real
    deadline prices/positions/clubs; the ``player_gw`` snapshot would
    resurrect last season's context for a live GW1.  Otherwise
    prices/positions are re-derived from the latest ``player_gw`` snapshot
    at/before ``start_gw`` (falling back to the previous season for players
    without rows yet); the prediction frames' context fills any player the
    snapshots miss.  Players with no resolvable price/position/team are
    dropped with a warning (the MILP cannot place them).
    """
    import numpy as np
    import pandas as pd

    context_cols = ["player_code", "price", "position", "team_code", "web_name"]
    base = (
        xp_window.sort_values("gw", kind="stable")
        .groupby("player_code", as_index=False)
        .head(1)
        .copy()
    )
    for col in context_cols[1:]:
        if col not in base.columns:
            base[col] = np.nan
    base = base[context_cols].reset_index(drop=True)

    # Secondary context: the per-fixture predictions frame (first fixture per player).
    if pred_fixture is not None and len(pred_fixture):
        fx = pred_fixture[
            (pred_fixture["season"] == season) & (pred_fixture["gw"] >= start_gw)
        ]
        have = [c for c in ("price", "position", "team_code") if c in fx.columns]
        if len(fx) and have:
            fxc = (
                fx.sort_values("gw", kind="stable")
                .groupby("player_code", as_index=False)
                .head(1)[["player_code", *have]]
            )
            base = base.merge(fxc, on="player_code", how="left", suffixes=("", "_fx"))
            for col in have:
                base[col] = base[col].where(base[col].notna(), base[f"{col}_fx"])
                base = base.drop(columns=f"{col}_fx")

    # Live primary source: the bootstrap roster snapshot for this season.
    roster_covers = False
    roster_path = processed / "live_roster.parquet"
    if roster_path.exists():
        roster = pd.read_parquet(roster_path)
        if "season" in roster.columns and (roster["season"] == season).any():
            snap = roster.loc[
                roster["season"] == season,
                ["player_code", "price", "position", "team_code", "web_name"],
            ].drop_duplicates("player_code")
            base = base.merge(snap, on="player_code", how="left", suffixes=("", "_lr"))
            for col in ("price", "position", "team_code", "web_name"):
                base[col] = base[f"{col}_lr"].where(base[f"{col}_lr"].notna(), base[col])
                base = base.drop(columns=f"{col}_lr")
            roster_covers = True
            console.print(
                f"  prices: live roster snapshot ({roster_path.name}) is the primary source"
            )

    # player_gw snapshot at/before start_gw: primary source pre-relaunch, gap
    # filler when the live roster covers the season (previous-season rows cover
    # players with no target-season appearance yet).
    pg_path = processed / "player_gw.parquet"
    if pg_path.exists():
        pg = pd.read_parquet(
            pg_path, columns=["season", "gw", "player_code", "team_code", "position", "value"]
        )
        mask = ((pg["season"] == season) & (pg["gw"] <= start_gw)) | (
            pg["season"] == season - 1
        )
        snap = (
            pg.loc[mask]
            .sort_values(["season", "gw"], kind="stable")
            .groupby("player_code", as_index=False)
            .tail(1)
            .rename(columns={"value": "price"})[["player_code", "price", "position", "team_code"]]
        )
        base = base.merge(snap, on="player_code", how="left", suffixes=("", "_pg"))
        for col in ("price", "position", "team_code"):
            if roster_covers:
                base[col] = base[col].where(base[col].notna(), base[f"{col}_pg"])
            else:
                base[col] = base[f"{col}_pg"].where(base[f"{col}_pg"].notna(), base[col])
            base = base.drop(columns=f"{col}_pg")

    # Names: fill gaps from the canonical players table when available.
    players_path = processed / "players.parquet"
    if base["web_name"].isna().any() and players_path.exists():
        players = pd.read_parquet(players_path, columns=["player_code", "web_name"])
        players = players.drop_duplicates("player_code")
        base = base.merge(players, on="player_code", how="left", suffixes=("", "_pl"))
        base["web_name"] = base["web_name"].where(
            base["web_name"].notna(), base["web_name_pl"]
        )
        base = base.drop(columns="web_name_pl")

    unresolved = (
        base["price"].isna() | base["position"].isna() | base["team_code"].isna()
    )
    if unresolved.any():
        console.print(
            f"  [yellow]{int(unresolved.sum())} player(s) have no resolvable "
            "price/position/team — excluded from the optimizer pool[/yellow]"
        )
        base = base.loc[~unresolved]
    if base.empty:
        raise ValueError("no players with resolvable prices — cannot optimize")
    base["player_code"] = base["player_code"].astype(int)
    base["price"] = base["price"].astype(float).round().astype(int)
    base["team_code"] = base["team_code"].astype(int)
    base["position"] = base["position"].astype(str)
    base["web_name"] = [
        str(n) if isinstance(n, str) and n else f"player {c}"
        for c, n in zip(base["player_code"], base["web_name"], strict=True)
    ]
    return base.reset_index(drop=True)


def _optimize_is_live(processed: Path, season: int, start_gw: int) -> bool:
    """True when the target window still has unplayed fixtures (live mode)."""
    import pandas as pd

    fx_path = processed / "fixtures.parquet"
    if not fx_path.exists():
        return False
    fx = pd.read_parquet(
        fx_path, columns=["season", "gw", "kickoff_utc", "finished", "void"]
    )
    window = fx[(fx["season"] == season) & (fx["gw"] >= start_gw) & ~fx["void"].astype(bool)]
    if window.empty:
        return False
    upcoming = window[
        ~window["finished"].astype(bool)
        & (window["kickoff_utc"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta("12h"))
    ]
    return not upcoming.empty


def _load_chip_sim_report(processed: Path, season: int, start_gw: int) -> Any | None:
    """The persisted season-simulation report, when it matches the target window.

    ``fplai simulate`` writes ``chip_sim_report.json`` beside ``chip_sim.parquet``; a
    later ``fplai optimize`` folds those verdicts into the recommendation when the
    report targets the same (season, next GW).  Stale or unreadable reports are
    skipped with a message — re-run ``fplai simulate`` after the deadline rolls over.
    """
    path = processed / "chip_sim_report.json"
    if not path.exists():
        return None
    from fplai.optimizer.season_sim import ChipSimReport

    try:
        report = ChipSimReport.model_validate_json(path.read_text())
    except Exception as exc:  # noqa: BLE001 - a stale artifact must not block optimize
        console.print(f"  [yellow]chip_sim_report.json unreadable ({exc}) — ignoring[/yellow]")
        return None
    if int(report.season) != season or int(report.window_first_gw) != start_gw:
        console.print(
            f"  [yellow]chip_sim_report.json targets season {report.season} "
            f"GW{report.window_first_gw} (want {season} GW{start_gw}) — stale, "
            "ignoring (re-run `fplai simulate`)[/yellow]"
        )
        return None
    console.print(
        f"  chip advice: folding season-simulation verdicts (GWs "
        f"{report.window_first_gw}-{report.window_last_gw}, "
        f"{report.n_rollouts:,} rollouts)"
    )
    return report


def _entry_state(entry_id: int, pred_gw: pd.DataFrame) -> Any | None:
    """Best-effort ``SquadState`` for ``entry_id``; None (with a warning) on failure.

    Falls back to None-state (initial-squad mode) when the API call fails or when
    the entry's ``(season, current_gw)`` has no prediction rows — the pre-launch
    demo optimizes a 2025-26 backtest window that a live entry cannot match.
    """
    from fplai.optimizer import state as state_mod

    try:
        state = state_mod.from_entry(entry_id)
    except Exception as exc:  # noqa: BLE001 - degrade to initial-squad mode, warned
        console.print(
            f"  [yellow]could not build squad state for entry {entry_id}: {exc} — "
            "falling back to initial-squad mode[/yellow]"
        )
        return None
    covered = (
        (pred_gw["season"] == state.season) & (pred_gw["gw"] == state.current_gw)
    ).any()
    if not covered:
        console.print(
            f"  [yellow]entry {entry_id} is at season {state.season} GW{state.current_gw} "
            "but the predictions on disk do not cover that GW — falling back to "
            "initial-squad (demo) mode[/yellow]"
        )
        return None
    return state


@contextmanager
def _capture_chip_curves() -> Iterator[dict[str, pd.DataFrame]]:
    """Capture the raw chip-EV frame computed inside ``build_recommendation``.

    ``fplai.optimizer.plans`` routes all chip-curve computation through its
    module-level ``chip_ev_curves`` proxy (a documented monkeypatch seam).  We
    temporarily wrap it so the pipeline can persist ``chip_curves.parquet``
    without a second (expensive) round of forced-chip re-solves.  Single-threaded
    CLI use only; the original proxy is always restored.
    """
    from fplai.optimizer import plans

    captured: dict[str, pd.DataFrame] = {}
    original = plans.chip_ev_curves

    def _wrapped(*args: Any, **kwargs: Any) -> pd.DataFrame:
        curves = original(*args, **kwargs)
        captured["curves"] = curves
        return curves

    plans.chip_ev_curves = _wrapped  # type: ignore[assignment]
    try:
        yield captured
    finally:
        plans.chip_ev_curves = original  # type: ignore[assignment]


def _squad_position_lines(
    codes: Sequence[int],
    names: dict[int, str],
    positions: dict[int, str],
    captain: int | None = None,
    vice: int | None = None,
) -> list[str]:
    """One display line per position group (GKP/DEF/MID/FWD) with (C)/(V) markers."""
    lines: list[str] = []
    for pos in ("GKP", "DEF", "MID", "FWD"):
        members = [c for c in codes if positions.get(c) == pos]
        if not members:
            continue
        parts = []
        for c in members:
            label = names.get(c, f"player {c}")
            if c == captain:
                label += " (C)"
            elif c == vice:
                label += " (V)"
            parts.append(label)
        lines.append(f"{pos}: " + ", ".join(parts))
    unknown = [c for c in codes if positions.get(c) not in ("GKP", "DEF", "MID", "FWD")]
    if unknown:
        lines.append("?: " + ", ".join(names.get(c, f"player {c}") for c in unknown))
    return lines


def _print_verdict(rec: Recommendation, prices: pd.DataFrame) -> None:
    """Rich-formatted weekly verdict: action, transfers, captain, chips, dream team."""
    from rich.table import Table

    from fplai import rules
    from fplai.optimizer.plans import CHIP_NAMES

    names = {int(c): str(n) for c, n in zip(prices["player_code"], prices["web_name"], strict=True)}
    positions = {
        int(c): str(p) for c, p in zip(prices["player_code"], prices["position"], strict=True)
    }
    price_of = {
        int(c): int(p) for c, p in zip(prices["player_code"], prices["price"], strict=True)
    }

    if rec.action == "initial-squad":
        headline = "build this initial squad (fresh £100.0m)"
    elif rec.action == "hold":
        headline = "hold — make no transfers"
    elif rec.action.startswith("chip:"):
        cid = rec.action.removeprefix("chip:")
        headline = f"play {CHIP_NAMES.get(cid[:2], cid)} ({cid})"
    else:
        n = len(rec.transfers)
        headline = f"make {n} transfer{'s' if n != 1 else ''}"
    console.print(
        f"\n[bold]verdict[/bold] — {rules.season_label(rec.season)} GW{rec.gw}: "
        f"[bold cyan]{headline}[/bold cyan]  "
        f"({rec.expected_points:.1f} xP this GW, plan objective {rec.objective:.1f})"
    )

    # In initial-squad mode every pick is a "transfer in" — the XI/squad display
    # below covers it, so the 15-row table would be noise.
    if rec.transfers and rec.action != "initial-squad":
        table = Table(title="transfers this GW", title_justify="left")
        for col in ("in", "out", "ΔxP (horizon)", "support"):
            table.add_column(col)
        for pair in rec.transfers:
            table.add_row(
                pair.player_in_name or "—",
                pair.player_out_name or "—",
                f"{pair.xp_delta:+.1f}",
                f"{pair.support_pct:.0f}%" if pair.support_pct is not None else "—",
            )
        console.print(table)
        if rec.hits:
            console.print(f"  hits: −{rec.hits} pts")

    if rec.captain is not None:
        vice = f"; vice {names.get(rec.vice, rec.vice)}" if rec.vice is not None else ""
        console.print(
            f"  captain: [bold]{names.get(rec.captain, rec.captain)}[/bold]{vice}"
        )
    if rec.lineup:
        console.print(f"  XI ({rec.formation or '?'}):")
        for line in _squad_position_lines(rec.lineup, names, positions, rec.captain, rec.vice):
            console.print(f"    {line}")
        bench = ", ".join(names.get(c, f"player {c}") for c in rec.bench_order)
        if bench:
            console.print(f"    bench: {bench}")
        cost = sum(price_of.get(c, 0) for c in rec.squad)
        if cost:
            console.print(f"    squad cost: £{cost / 10:.1f}m")

    if rec.chip_advice:
        table = Table(title="chip advice", title_justify="left")
        for col in ("chip", "verdict", "planned GW", "EV now", "best GW", "best EV"):
            table.add_column(col)
        for adv in rec.chip_advice:
            table.add_row(
                adv.chip,
                adv.verdict,
                str(adv.planned_gw) if adv.planned_gw is not None else "—",
                f"{adv.ev_now:+.1f}" if adv.ev_now is not None else "—",
                str(adv.best_gw) if adv.best_gw is not None else "—",
                f"{adv.best_ev:+.1f}" if adv.best_ev is not None else "—",
            )
        console.print(table)

    dream = rec.dream_team
    if dream is not None and rec.action != "initial-squad":
        console.print(
            f"  dream team (fresh £100m benchmark, GW{dream.gw}): "
            f"{dream.expected_points:.1f} xP, {dream.formation or '?'}, "
            f"£{dream.total_cost / 10:.1f}m"
        )
        for line in _squad_position_lines(
            dream.lineup, names, positions, dream.captain, dream.vice
        ):
            console.print(f"    {line}")

    if rec.rationale:
        console.print("  [bold]why[/bold]:")
        for bullet in rec.rationale[:6]:
            console.print(f"    • {bullet}")


def run_optimize(
    entry_id: int | None = None,
    season: int | None = None,
    gw: int | None = None,
    *,
    horizon: int | None = None,
    run_chips: bool = True,
    run_stability: bool = True,
    stability_n: int = 30,
    processed_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Recommendation:
    """Run the squad optimizer on the predictions on disk and write the verdict.

    Loads ``predictions_gw.parquet`` (+ ``predictions.parquet`` for per-fixture
    context), re-derives the optimizer's prices/positions from the latest
    ``player_gw`` snapshot, builds the :class:`~fplai.optimizer.state.SquadState`
    (``entry_id`` argument, else ``FPLAI_ENTRY_ID``; None-state initial-squad
    mode otherwise) and runs :func:`fplai.optimizer.plans.build_recommendation`
    with ``horizon = min(settings.horizon_gws, available GWs)``.

    Pre-launch demo mode: while 2026-27 is not live there are no upcoming
    fixtures, so the predictions window on disk is a 2025-26 backtest
    (``fplai predict --season 2025 --gw N``); ``--season``/``--gw`` select
    within it and the verdict is the demo the dashboard shows until launch.

    Artifacts (served by the API): ``recommendation.json`` (full
    :class:`Recommendation`), ``dream_team.json`` (standalone benchmark squad)
    and ``chip_curves.parquet`` (raw chip EV curves, when computed).

    Args:
        entry_id: FPL entry (team) id; falls back to ``settings.entry_id``
            (0 = unset -> None-state initial-squad mode).
        season/gw: Select this backtest window from the predictions on disk
            (both or neither).
        horizon: Planning horizon cap (default ``settings.horizon_gws``);
            always clipped to the GWs available in the predictions.
        run_chips/run_stability/stability_n: Forwarded to
            ``build_recommendation`` (chip EV curves and noise re-solves are
            the slow parts — disable for a fast verdict).
        processed_dir/out_dir: Input/output overrides (default
            ``config.PROCESSED_DIR``; ``out_dir`` defaults to the input dir).

    Returns:
        The :class:`~fplai.optimizer.plans.Recommendation` (also written to
        ``recommendation.json``).

    Raises:
        FileNotFoundError: when no predictions exist on disk.
        ValueError: when the requested season/GW window has no prediction rows.
    """
    import pandas as pd

    from fplai import config
    from fplai.optimizer import plans

    if (season is None) != (gw is None):
        raise ValueError("season and gw must be given together (demo mode) or not at all")

    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    out = Path(out_dir) if out_dir is not None else processed

    gw_path = processed / "predictions_gw.parquet"
    if not gw_path.exists():
        raise FileNotFoundError(
            f"{gw_path} is missing — run `fplai predict` first (pre-launch: "
            "`fplai predict --season 2025 --gw 34` for the demo window)"
        )
    pred_gw = pd.read_parquet(gw_path)
    if pred_gw.empty:
        raise ValueError(f"{gw_path} is empty — re-run `fplai predict`")
    fx_path = processed / "predictions.parquet"
    pred_fx = pd.read_parquet(fx_path) if fx_path.exists() else None

    # ---- target window selection -------------------------------------------------
    if season is not None and gw is not None:
        target_season, from_gw = int(season), int(gw)
        if not ((pred_gw["season"] == target_season) & (pred_gw["gw"] >= from_gw)).any():
            raise ValueError(
                f"predictions_gw.parquet has no rows for season {target_season} "
                f"GW>={from_gw} — run `fplai predict --season {target_season} "
                f"--gw {from_gw}` first"
            )
    else:
        target_season = int(pred_gw["season"].max())
        from_gw = int(pred_gw.loc[pred_gw["season"] == target_season, "gw"].min())

    # ---- squad state (entry vs None-state initial squad) -------------------------
    resolved_entry = entry_id if entry_id is not None else (config.settings.entry_id or None)
    state = _entry_state(int(resolved_entry), pred_gw) if resolved_entry else None
    if state is not None:
        target_season, from_gw = int(state.season), int(state.current_gw)

    xp = pred_gw[(pred_gw["season"] == target_season) & (pred_gw["gw"] >= from_gw)].copy()
    available_gws = sorted(int(g) for g in xp["gw"].unique())
    start_gw = available_gws[0]
    horizon_cap = horizon if horizon is not None else config.settings.horizon_gws
    plan_horizon = max(1, min(horizon_cap, len(available_gws)))

    live = _optimize_is_live(processed, target_season, start_gw)
    prices = _optimizer_prices(xp, processed, target_season, start_gw, pred_fx)

    mode = "live" if live else "pre-launch demo (backtest predictions)"
    state_desc = f"entry {resolved_entry}" if state is not None else "none (initial squad)"
    console.print(
        f"[bold]optimize[/bold]: {mode} — season {target_season} GW{start_gw}, "
        f"horizon {plan_horizon} (GWs {available_gws[:plan_horizon]}), "
        f"{len(prices):,} players, state: {state_desc}"
    )
    if not live:
        console.print(
            "  [yellow]demo mode: optimizing a 2025-26 backtest window — this is what "
            "the dashboard shows until the 2026-27 game launches[/yellow]"
        )

    chip_sim = _load_chip_sim_report(processed, target_season, start_gw)
    as_of = dt.datetime.now(dt.UTC)

    # Execution-timing inputs (advisory rationale only, never solve inputs):
    # official price-predictor progress per player + the real deadline timestamp.
    price_trends: dict[int, float] = {}
    trend_path = processed / "live_roster.parquet"
    if trend_path.exists():
        trend_roster = pd.read_parquet(trend_path)
        if "price_change_percent" in trend_roster.columns:
            rows_t = trend_roster[trend_roster["season"] == target_season]
            price_trends = {
                int(c): float(v)
                for c, v in zip(rows_t["player_code"], rows_t["price_change_percent"])
                if pd.notna(v) and float(v) != 0.0
            }
    from fplai.publish import _season_state

    snap_state = _season_state(config.RAW_DIR)
    next_deadline_utc = (
        snap_state.next_deadline_utc
        if snap_state is not None
        and snap_state.season == target_season
        and snap_state.next_gw == start_gw
        else None
    )
    # Tail guard (research follow-up, community-solver practice ≈2): transfers
    # scheduled in the horizon's final GWs monetize almost none of their value
    # inside the window — ban them there (WC/FH chip weeks stay exempt). Only
    # applied when the horizon leaves at least one actionable GW ahead of the tail.
    from fplai.optimizer.milp import SolveParams

    tail = 2 if plan_horizon >= 4 else 0
    solve_params = SolveParams(no_transfer_last_gws=tail)
    with _capture_chip_curves() as captured:
        rec = plans.build_recommendation(
            state,
            xp,
            prices,
            as_of=as_of,
            horizon=plan_horizon,
            stability_n=stability_n,
            run_chips=run_chips,
            run_stability=run_stability,
            chip_sim=chip_sim,
            params=solve_params,
            price_trends=price_trends or None,
            next_deadline_utc=next_deadline_utc,
        )

    # ---- degenerate-solve gate ---------------------------------------------------
    # A healthy squad's first-GW expected points can't collapse far below what the
    # prediction frame obviously supports (observed failure: a timeout incumbent =
    # cheapest legal squad at 12.2 xP while top players summed to 60+). Compare
    # against a cheap proxy bound — the sum of the 11 highest xp values in the
    # first planned GW — and refuse to publish artifacts under 40% of it.
    first_gw_xp = xp.loc[xp["gw"] == start_gw, "xp"]
    if len(first_gw_xp) >= 11:
        proxy_bound = float(first_gw_xp.nlargest(11).sum())
        if rec.expected_points < 0.4 * proxy_bound:
            raise RuntimeError(
                f"optimizer returned a degenerate plan (GW{start_gw} expected "
                f"{rec.expected_points:.1f} xP vs ~{proxy_bound:.0f} attainable) — "
                "artifacts NOT overwritten; previous recommendation stays live. "
                "This usually means the MILP timed out on a trivial incumbent; "
                "re-run with a longer time limit."
            )

    # ---- artifacts (the API serves these files verbatim) -------------------------
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "recommendation.json"
    if state is not None and resolved_entry:
        # Entry-scoped verdict: the wrapped shape lets GET /api/my-team/{entry_id}
        # serve this file directly (it matches on the entry_id key); GET
        # /api/recommendation unwraps the "recommendation" key transparently.
        rec_path.write_text(
            json.dumps(
                {
                    "entry_id": int(resolved_entry),
                    "squad_state": json.loads(state.model_dump_json()),
                    "recommendation": json.loads(rec.model_dump_json()),
                },
                indent=2,
            )
        )
    else:
        rec_path.write_text(rec.model_dump_json(indent=2))
    written = [rec_path]
    if rec.dream_team is not None:
        dream_path = out / "dream_team.json"
        dream_path.write_text(rec.dream_team.model_dump_json(indent=2))
        written.append(dream_path)
    # chip_curves.parquet: the season-simulation v2 frame (full chip window + MC
    # probability columns) when a fresh matching report exists, else the captured
    # v1 forced-chip curves from build_recommendation's horizon re-solves.
    curves = chip_sim.to_frame() if chip_sim is not None else captured.get("curves")
    if curves is not None and len(curves):
        curves_path = out / "chip_curves.parquet"
        curves.to_parquet(curves_path, index=False)
        written.append(curves_path)

    _print_verdict(rec, prices)
    console.print(
        "[bold]optimize[/bold]: wrote " + ", ".join(str(p) for p in written)
    )
    return rec


# ---------------------------------------------------------------------------
# simulate (stage 6: Monte Carlo season-simulation chip planner)
# ---------------------------------------------------------------------------


def run_simulate(
    entry_id: int | None = None,
    *,
    rollouts: int = 1000,
    seed: int = 0,
    through_gw: int | None = None,
    processed_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Any:
    """Run the Monte Carlo chip-timing simulation and persist its verdicts.

    Runs :func:`fplai.optimizer.season_sim.simulate_chip_plans` over the FULL
    remaining chip window (next GW .. ``through_gw``, default the window end from
    ``rules.chip_windows`` — GW19 for set-1 chips), fixing the greedy
    chip-within-horizon problem: every chip placement is priced against holding,
    with rollout-level uncertainty from ``models.sampler.PointsSampler``.

    Artifacts (``out_dir``, default the processed dir):

    * ``chip_sim.parquet`` — the chip_curves-v2 frame (``ChipSimReport.to_frame()``).
    * ``chip_sim_report.json`` — the full report; ``run_optimize`` folds it into
      future recommendations while it stays fresh.
    * ``chip_curves.parquet`` — overwritten with the v2 frame (the dashboard's
      chip panel source; additive columns, v1 consumers keep working).
    * ``recommendation.json`` — when present and targeting the same (season, GW),
      its ``chip_advice``/rationale are re-verdicted via
      :func:`fplai.optimizer.plans.apply_sim_to_recommendation`.

    Args:
        entry_id: FPL entry id (falls back to ``settings.entry_id``; unset ->
            None-state initial-squad mode, matching ``run_optimize``).
        rollouts: Monte Carlo rollouts (default 1000).
        seed: Sampler seed (deterministic report for fixed inputs + seed).
        through_gw: Last window GW (default: the current chip window's end).
        processed_dir/out_dir: Input/output overrides.

    Returns:
        The :class:`~fplai.optimizer.season_sim.ChipSimReport`.

    Raises:
        FileNotFoundError: when predictions are missing.
        ValueError: when predictions do not cover the window — run
            ``fplai predict --through-gw N`` first.
    """
    import pandas as pd

    from fplai import config
    from fplai.models.sampler import chip_window_end
    from fplai.optimizer import plans
    from fplai.optimizer.season_sim import simulate_chip_plans

    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    out = Path(out_dir) if out_dir is not None else processed

    fx_path = processed / "predictions.parquet"
    gw_path = processed / "predictions_gw.parquet"
    for path in (fx_path, gw_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — run `fplai predict --through-gw 19` first"
            )
    pred_fx = pd.read_parquet(fx_path)
    pred_gw = pd.read_parquet(gw_path)
    if pred_fx.empty or pred_gw.empty:
        raise ValueError("predictions on disk are empty — re-run `fplai predict`")

    target_season = int(pred_gw["season"].max())
    start_gw = int(pred_gw.loc[pred_gw["season"] == target_season, "gw"].min())
    resolved_entry = entry_id if entry_id is not None else (config.settings.entry_id or None)
    state = _entry_state(int(resolved_entry), pred_gw) if resolved_entry else None
    if state is not None:
        target_season, start_gw = int(state.season), int(state.current_gw)

    window_end = through_gw if through_gw is not None else chip_window_end(
        target_season, start_gw
    )
    if window_end < start_gw:
        raise ValueError(f"through_gw {window_end} is before the next GW {start_gw}")
    window = range(start_gw, window_end + 1)

    xp_win = pred_gw[
        (pred_gw["season"] == target_season) & (pred_gw["gw"] >= start_gw)
    ].copy()
    prices = _optimizer_prices(xp_win, processed, target_season, start_gw, pred_fx)

    state_desc = f"entry {resolved_entry}" if state is not None else "none (initial squad)"
    console.print(
        f"[bold]simulate[/bold]: season {target_season} chip window GWs "
        f"{start_gw}-{window_end}, {rollouts:,} rollouts, seed {seed}, "
        f"{len(prices):,} players, state: {state_desc}"
    )

    report = simulate_chip_plans(
        pred_fx,
        prices,
        state,
        window=window,
        n_rollouts=rollouts,
        seed=seed,
    )

    bq = report.backbone
    console.print(
        f"  backbone: objective {bq.objective:.1f}, gap {bq.gap:.2%}, "
        f"{bq.solve_seconds:.0f}s ({bq.status}); {report.n_segment_solves} segment "
        f"solves; appearance via {report.appearance_source}; "
        f"total {report.sim_seconds:.0f}s"
    )
    from rich.table import Table

    table = Table(title="chip verdicts (Monte Carlo)", title_justify="left")
    for col in ("chip", "verdict now", "best GW", "E[gain]", "sd", "P(best)", "P(now beats hold)"):
        table.add_column(col)
    for v in report.verdicts:
        now_stat = next(
            (s for s in report.stats if s.chip == v.chip and s.gw == start_gw and s.evaluated),
            None,
        )
        p_now = now_stat.p_beats_hold if now_stat is not None else None
        table.add_row(
            v.chip,
            ("play" if p_now is not None and p_now > 0.5 else "hold")
            if v.recommended_gw is not None
            else "—",
            str(v.recommended_gw) if v.recommended_gw is not None else "—",
            f"{v.e_gain:+.1f}" if v.e_gain is not None else "—",
            f"{v.sd:.1f}" if v.sd is not None else "—",
            f"{v.confidence:.2f}" if v.confidence is not None else "—",
            f"{p_now:.2f}" if p_now is not None else "—",
        )
    console.print(table)
    if report.schedules:
        top = report.schedules[0]
        placed = ", ".join(f"{c}@GW{g}" for c, g in sorted(top.placements.items()))
        console.print(
            f"  best joint schedule: {placed or 'play nothing'} "
            f"(E {top.e_total_gain:+.1f}, P(best) {top.p_best:.2f})"
        )

    # ---- artifacts -------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    frame = report.to_frame()
    written = []
    for name in ("chip_sim.parquet", "chip_curves.parquet"):
        path = out / name
        frame.to_parquet(path, index=False)
        written.append(path)
    report_path = out / "chip_sim_report.json"
    report_path.write_text(report.model_dump_json(indent=1))
    written.append(report_path)

    # ---- fold the verdicts into an existing recommendation.json ----------------------
    rec_path = out / "recommendation.json"
    if rec_path.exists():
        try:
            raw = json.loads(rec_path.read_text())
            wrapped = "recommendation" in raw and "entry_id" in raw
            rec = plans.Recommendation.model_validate(
                raw["recommendation"] if wrapped else raw
            )
            if int(rec.season) == target_season and int(rec.gw) == start_gw:
                updated = plans.apply_sim_to_recommendation(rec, report)
                if wrapped:
                    raw["recommendation"] = json.loads(updated.model_dump_json())
                    rec_path.write_text(json.dumps(raw, indent=2))
                else:
                    rec_path.write_text(updated.model_dump_json(indent=2))
                written.append(rec_path)
                console.print("  recommendation.json chip advice re-verdicted from the sim")
            else:
                console.print(
                    f"  [yellow]recommendation.json targets season {rec.season} GW{rec.gw} "
                    f"(sim: {target_season} GW{start_gw}) — not folding; re-run "
                    "`fplai optimize`[/yellow]"
                )
        except Exception as exc:  # noqa: BLE001 - the sim artifacts stand on their own
            console.print(
                f"  [yellow]could not fold verdicts into recommendation.json: {exc}[/yellow]"
            )

    console.print(
        "[bold]simulate[/bold]: wrote " + ", ".join(str(p) for p in written)
    )
    return report


# ---------------------------------------------------------------------------
# backtest (stage 4 harness)
# ---------------------------------------------------------------------------


def run_backtest(season: int, gws: Sequence[int] | None = None) -> Any:
    """Run the stage-4 walk-forward backtest harness over one season.

    Thin wrapper around ``fplai.backtest.harness.run(season=..., gws=...)`` —
    the documented stage-4 contract (per-GW walk-forward loop with squad-state
    rolling and season-points policy eval).  The harness module is built by the
    backtest task; until it lands this raises a clear error instead of an
    ImportError traceback.

    Args:
        season: Season start year to backtest (e.g. 2025).
        gws: Optional GW subset (default: the harness's full-season default).

    Raises:
        RuntimeError: when ``fplai.backtest.harness`` is not available yet.
    """
    try:
        from fplai.backtest import harness
    except ImportError as exc:
        raise RuntimeError(
            "the backtest harness (fplai.backtest.harness) is not available yet — "
            "stage 4 is pending; see docs/ARCHITECTURE.md. "
            f"(import error: {exc})"
        ) from exc
    gw_note = f" GWs {list(gws)}" if gws else ""
    console.print(f"[bold]backtest[/bold]: season {season}{gw_note}")
    return harness.run(season=season, gws=list(gws) if gws is not None else None)


# ---------------------------------------------------------------------------
# publish-static (the GitHub-Pages data bundle)
# ---------------------------------------------------------------------------


def run_publish_static(out_dir: Path | None = None) -> Any:
    """Publish the static site-data bundle from ``data/processed/`` artifacts.

    Thin wrapper around :func:`fplai.publish.build_bundle` (the shared static-data
    contract the GitHub-Pages frontend consumes).  ``meta.json``'s deadline comes
    from the newest raw FPL snapshot and ``model_version`` from the models
    manifest; per-file byte sizes are printed and returned.

    Args:
        out_dir: Bundle directory (default ``<repo>/site-data``, gitignored).

    Returns:
        The :class:`fplai.publish.BundleReport`.

    Raises:
        FileNotFoundError: when required artifacts (predictions, roster) are missing.
        ValueError: when the bundle exceeds the size budget or artifacts are
            mutually inconsistent.
    """
    from fplai import config
    from fplai.publish import MAX_BUNDLE_BYTES, build_bundle

    out = Path(out_dir) if out_dir is not None else config.REPO_ROOT / "site-data"
    console.print(f"[bold]publish-static[/bold]: {config.PROCESSED_DIR} -> {out}")
    report = build_bundle(
        config.PROCESSED_DIR, out, raw_dir=config.RAW_DIR, models_dir=config.MODELS_DIR
    )
    for name, size in report.files.items():
        console.print(f"  {name:<22} {size:>10,} B")
    console.print(
        f"  [bold]total[/bold] {report.total_bytes:>21,} B "
        f"(budget {MAX_BUNDLE_BYTES:,} B) — season {report.season}, "
        f"GW{report.window[0]}..{report.window[1]}"
    )
    if report.missing:
        console.print(f"  [yellow]missing optional artifacts:[/yellow] {', '.join(report.missing)}")
    return report
