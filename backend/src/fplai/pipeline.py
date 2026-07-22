"""End-to-end pipeline orchestration: snapshot -> backfill/pulls -> build -> models.

Each stage is implemented in its own module; this file only sequences them and
is the single place the CLI and API call into.

Data stages (snapshot / backfill / build / the data portion of refresh) and the
model stages (train / predict) are fully wired; optimize is a stub until the
optimizer is threaded through ``refresh``.

Train/predict artifacts live under ``config.MODELS_DIR`` with a
``manifest.json`` describing versions, the training window and headline
metrics.  ``run_predict`` supports two modes:

* **live** (default): predict every upcoming fixture in the next ``horizon``
  gameweeks.  Rows for unplayed fixtures are synthesized from each player's
  most recent appearance (team/position/price roll forward).  When no upcoming
  fixtures exist on disk (e.g. between seasons) it degrades gracefully to a
  message and no output.
* **backtest** (``season`` + ``gw`` given): predict historical GWs
  ``gw .. gw+horizon-1`` of ``season`` from strictly-prior information — the
  walk-forward primitive.  Features for each target GW use only matches before
  that GW (enforced by ``features/windows.py``); models should be trained with
  a matching ``before_season``/``before_gw`` cutoff for a leakage-clean eval.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    import pandas as pd

    from fplai.data.fpl_api import SeasonState
    from fplai.models.bonus import BonusCalibration

console = Console()
logger = logging.getLogger(__name__)

#: Processed tables loaded by :func:`load_processed_tables` (odds is optional).
_PROCESSED_TABLES = ("players", "teams", "fixtures", "player_match", "player_gw")


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
    if spec is None or not spec.strip():
        return None
    seasons: set[int] = set()
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ".." in entry:
            lo_s, _, hi_s = entry.partition("..")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"bad season range {entry!r} (want e.g. 2016..2025)") from exc
            if lo > hi:
                raise ValueError(f"bad season range {entry!r}: start > end")
            seasons.update(range(lo, hi + 1))
        else:
            try:
                seasons.add(int(entry))
            except ValueError as exc:
                raise ValueError(f"bad season {entry!r} (want e.g. 2024 or 2016..2025)") from exc
    for s in seasons:
        if not 2016 <= s <= 2026:
            raise ValueError(f"season {s} out of supported range 2016..2026")
    return sorted(seasons)


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


def run_refresh() -> None:
    """Pull fresh data from all sources, rebuild tables, then the model stages.

    The data portion (snapshot -> incremental pulls -> build) is fully wired
    and must succeed; the model stages are best-effort so `fplai refresh`
    stays exit-0: train is never run automatically (run `fplai train`),
    predict runs when artifacts exist, optimize is not yet wired here.
    """
    state = run_snapshot()
    console.print("[bold]refresh[/bold]: incremental pulls")
    _refresh_pulls(state)
    run_build()
    from fplai import config

    if (config.MODELS_DIR / "manifest.json").exists():
        console.print("[bold]train[/bold]: using existing artifacts (retrain with `fplai train`)")
        try:
            run_predict()
        except Exception as exc:  # noqa: BLE001 - refresh must stay exit-0
            console.print(f"[bold]predict[/bold]: [yellow]failed: {exc}[/yellow] (skipping)")
    else:
        console.print("[bold]train[/bold]: no model artifacts yet — run `fplai train` (skipping)")
        console.print("[bold]predict[/bold]: skipped (no model artifacts)")
    console.print("[bold]optimize[/bold]: not yet wired into refresh (skipping)")
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


def _future_player_match(pm: pd.DataFrame, target_fx: pd.DataFrame) -> pd.DataFrame:
    """Synthesize player_match rows for unplayed fixtures (live-mode prediction).

    Rosters roll forward from each player's most recent appearance: a player
    belongs to the squad of the team of their latest ``player_match`` row,
    keeping that row's position/price/element id.  Only players seen in the
    target season or the season before are considered (no resurrecting
    long-departed players).  All outcome columns are NaN — the feature builder
    skips NaNs, so these rows contribute nothing to anyone's history.
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run minutes/team/rates + assemble over one already-selected feature frame."""
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
        except Exception as exc:  # noqa: BLE001 - odds are optional enrichment
            console.print(f"  [yellow]odds blend skipped: {exc}[/yellow]")

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
    return pred, gw_pred


def run_predict(
    season: int | None = None,
    gw: int | None = None,
    *,
    horizon: int | None = None,
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

    Returns ``{"predictions": path, "predictions_gw": path}`` or ``None``.
    """
    import pandas as pd

    from fplai import config
    from fplai.features.windows import build_feature_frame

    if (season is None) != (gw is None):
        raise ValueError("season and gw must be given together (backtest mode) or not at all")
    horizon = horizon if horizon is not None else config.settings.horizon_gws
    out = Path(out_dir) if out_dir is not None else config.PROCESSED_DIR

    tables = load_processed_tables()
    fixtures = tables["fixtures"]

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
        next_keys = (
            upcoming.groupby(["season", "gw"], as_index=False)["kickoff_utc"]
            .min()
            .sort_values("kickoff_utc")
            .head(horizon)[["season", "gw"]]
        )
        target_fx = upcoming.merge(next_keys, on=["season", "gw"])
        mode = (
            f"live: {len(target_fx)} upcoming fixtures over GWs "
            f"{sorted(next_keys['gw'].tolist())}"
        )
        pm = tables["player_match"]
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
    pred, gw_pred = _predict_frames(
        rows, fixtures, tables, (minutes_model, team_model, rates_model, calibration), use_odds
    )
    paths = {
        "predictions": out / "predictions.parquet",
        "predictions_gw": out / "predictions_gw.parquet",
    }
    pred.to_parquet(paths["predictions"], index=False)
    gw_pred.to_parquet(paths["predictions_gw"], index=False)

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


def run_optimize() -> None:
    """Run the squad optimizer and chip planner (stage 3 — not wired yet)."""
    console.print("[bold]optimize[/bold]: optimization not yet wired")
    raise SystemExit(1)
