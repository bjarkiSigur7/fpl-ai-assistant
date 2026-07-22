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
def train() -> None:
    """Retrain all models on the full historical dataset."""
    from fplai.pipeline import run_train

    run_train()


@app.command()
def predict() -> None:
    """Generate expected-points predictions for upcoming gameweeks."""
    from fplai.pipeline import run_predict

    run_predict()


@app.command()
def optimize() -> None:
    """Run the squad optimizer and chip planner on current predictions."""
    from fplai.pipeline import run_optimize

    run_optimize()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
