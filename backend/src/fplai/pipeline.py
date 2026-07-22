"""End-to-end pipeline orchestration: refresh -> train -> predict -> optimize.

Each stage is implemented in its own module; this file only sequences them and
is the single place the CLI and API call into.
"""

from rich.console import Console

console = Console()


def run_refresh() -> None:
    console.print("[bold]refresh[/bold]: data ingestion not yet implemented")
    raise SystemExit(1)


def run_train() -> None:
    console.print("[bold]train[/bold]: model training not yet implemented")
    raise SystemExit(1)


def run_predict() -> None:
    console.print("[bold]predict[/bold]: prediction not yet implemented")
    raise SystemExit(1)


def run_optimize() -> None:
    console.print("[bold]optimize[/bold]: optimization not yet implemented")
    raise SystemExit(1)
