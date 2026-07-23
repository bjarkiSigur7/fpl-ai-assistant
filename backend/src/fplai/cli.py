"""fplai command-line interface.

Thin typer wrappers over :mod:`fplai.pipeline`: data commands (snapshot,
backfill, build, refresh), model commands (train, predict), the optimizer
(optimize), the Monte Carlo chip-timing simulation (simulate) and the
walk-forward backtest harness (backtest).
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
    through_gw: Annotated[
        int | None,
        typer.Option(
            "--through-gw",
            help="Live mode: predict every GW through this one inclusive (e.g. 19 = the "
            "set-1 chip window end, what `fplai simulate` needs). Excludes --horizon.",
        ),
    ] = None,
    no_odds: Annotated[
        bool, typer.Option("--no-odds", help="Skip the bookmaker-odds blend.")
    ] = False,
) -> None:
    """Generate expected-points predictions (live fixtures, or --season/--gw backtest)."""
    from fplai.pipeline import run_predict

    if (season is None) != (gw is None):
        raise typer.BadParameter("--season and --gw must be given together")
    try:
        run_predict(
            season, gw, horizon=horizon, through_gw=through_gw, use_odds=not no_odds
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def optimize(
    entry_id: Annotated[
        int | None,
        typer.Option(
            "--entry-id",
            help="FPL entry (team) id to plan for; default FPLAI_ENTRY_ID from .env. "
            "Unset -> None-state initial-squad build.",
        ),
    ] = None,
    season: Annotated[
        int | None,
        typer.Option(
            help="Demo/backtest mode: season of the predictions window to optimize "
            "(pre-launch: the 2025-26 demo the dashboard shows)."
        ),
    ] = None,
    gw: Annotated[
        int | None,
        typer.Option(help="Demo/backtest mode: first GW of the window (needs --season)."),
    ] = None,
    horizon: Annotated[
        int | None,
        typer.Option(
            help="Planning horizon in GWs (default: settings; clipped to available "
            "predictions)."
        ),
    ] = None,
    no_chips: Annotated[
        bool, typer.Option("--no-chips", help="Skip the chip EV curves (faster).")
    ] = False,
    no_stability: Annotated[
        bool,
        typer.Option("--no-stability", help="Skip the plan-stability noise re-solves (faster)."),
    ] = False,
    stability_n: Annotated[
        int, typer.Option(help="Number of noise re-solves for plan stability.")
    ] = 30,
) -> None:
    """Run the squad optimizer on current predictions and write recommendation.json."""
    from fplai.pipeline import run_optimize

    if (season is None) != (gw is None):
        raise typer.BadParameter("--season and --gw must be given together")
    try:
        run_optimize(
            entry_id,
            season,
            gw,
            horizon=horizon,
            run_chips=not no_chips,
            run_stability=not no_stability,
            stability_n=stability_n,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]optimize failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def simulate(
    entry_id: Annotated[
        int | None,
        typer.Option(
            "--entry-id",
            help="FPL entry (team) id to simulate for; default FPLAI_ENTRY_ID from .env. "
            "Unset -> None-state initial-squad mode.",
        ),
    ] = None,
    rollouts: Annotated[
        int, typer.Option("--rollouts", help="Monte Carlo rollouts (default 1000).")
    ] = 1000,
    seed: Annotated[int, typer.Option(help="Sampler seed (deterministic report).")] = 0,
    through_gw: Annotated[
        int | None,
        typer.Option(
            "--through-gw",
            help="Last window GW (default: the current chip window's end, e.g. 19). "
            "Predictions must cover it — run `fplai predict --through-gw` first.",
        ),
    ] = None,
) -> None:
    """Monte Carlo chip-timing simulation over the full chip window (stage 6).

    Writes chip_sim.parquet + chip_sim_report.json, refreshes chip_curves.parquet
    with the v2 probability columns and re-verdicts recommendation.json's chip
    advice (play/hold with quantified hold value).
    """
    from fplai.pipeline import run_simulate

    try:
        run_simulate(entry_id, rollouts=rollouts, seed=seed, through_gw=through_gw)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]simulate failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def backtest(
    season: Annotated[
        int, typer.Option(help="Season start year to backtest, e.g. 2025.")
    ],
    gws: Annotated[
        str | None,
        typer.Option(
            help="GWs to backtest: single values and/or inclusive ranges, e.g. '30..38'. "
            "Default: the harness's full-season default."
        ),
    ] = None,
) -> None:
    """Run the walk-forward backtest harness (stage 4) over one season."""
    from fplai.pipeline import parse_gws, run_backtest

    try:
        gw_list = parse_gws(gws)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gws") from exc
    try:
        run_backtest(season, gw_list)
    except RuntimeError as exc:
        console.print(f"[red]backtest failed:[/red] {exc}")
        raise typer.Exit(1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
