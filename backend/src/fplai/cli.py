"""fplai command-line interface.

Commands are thin wrappers over the pipeline modules so `make refresh` etc. work
from day one; each fills in as the corresponding module lands.
"""

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()


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
