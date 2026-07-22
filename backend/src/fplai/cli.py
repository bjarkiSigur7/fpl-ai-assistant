"""fplai command-line interface.

Thin typer wrappers over :mod:`fplai.pipeline`.  Data commands (snapshot,
backfill, build, refresh) are fully wired; model commands (train, predict,
optimize) are stubs until stage 2+ modules land.
"""

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()

SeasonsOpt = Annotated[
    str | None,
    typer.Option(
        "--seasons",
        help="Season start years: single years and/or inclusive ranges, "
        "e.g. '2024,2025' or '2016..2025'. Default: all available.",
    ),
]


def _parse_seasons_opt(spec: str | None) -> list[int] | None:
    from fplai.pipeline import parse_seasons

    try:
        return parse_seasons(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--seasons") from exc


@app.command()
def snapshot() -> None:
    """Archive today's FPL API bootstrap+fixtures and print season state."""
    from fplai.pipeline import run_snapshot

    run_snapshot()


@app.command()
def backfill(seasons: SeasonsOpt = None) -> None:
    """Download historical raw data (vaastav, football-data, Understat, ClubElo)."""
    from fplai.pipeline import run_backfill

    run_backfill(_parse_seasons_opt(seasons))


@app.command()
def build(seasons: SeasonsOpt = None) -> None:
    """Build the processed parquet tables from raw data already on disk."""
    from fplai.pipeline import run_build

    run_build(_parse_seasons_opt(seasons))


@app.command()
def refresh() -> None:
    """Pull fresh data from all sources, then re-predict and re-optimize."""
    from fplai.pipeline import run_refresh

    run_refresh()


@app.command()
def train(
    seasons: SeasonsOpt = None,
    before_season: Annotated[
        int | None,
        typer.Option(
            help="Walk-forward cutoff: train strictly before this season "
            "(with --before-gw: before that GW of this season)."
        ),
    ] = None,
    before_gw: Annotated[
        int | None,
        typer.Option(help="Walk-forward cutoff GW within --before-season."),
    ] = None,
) -> None:
    """Retrain all models on the full historical dataset and save artifacts."""
    from fplai.pipeline import run_train

    if before_gw is not None and before_season is None:
        raise typer.BadParameter("--before-gw requires --before-season")
    run_train(
        _parse_seasons_opt(seasons), before_season=before_season, before_gw=before_gw
    )


@app.command()
def predict(
    season: Annotated[
        int | None,
        typer.Option(help="Backtest mode: season start year of the first GW to predict."),
    ] = None,
    gw: Annotated[
        int | None,
        typer.Option(help="Backtest mode: first GW to predict (needs --season)."),
    ] = None,
    horizon: Annotated[
        int | None, typer.Option(help="Number of GWs to predict (default: settings).")
    ] = None,
    no_odds: Annotated[
        bool, typer.Option("--no-odds", help="Skip the bookmaker-odds blend.")
    ] = False,
) -> None:
    """Generate expected-points predictions (live fixtures, or --season/--gw backtest)."""
    from fplai.pipeline import run_predict

    if (season is None) != (gw is None):
        raise typer.BadParameter("--season and --gw must be given together")
    run_predict(season, gw, horizon=horizon, use_odds=not no_odds)


@app.command()
def optimize() -> None:
    """Run the squad optimizer and chip planner on current predictions."""
    from fplai.pipeline import run_optimize

    run_optimize()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
