"""``nfl`` command-line interface.

All commands are stubs at this stage of the project; they exist so the CLI
surface, the ``/predict`` slash command and CI wiring can be exercised before
the pipeline lands.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from nfl import __version__

app = typer.Typer(
    name="nfl",
    help="NFL in-game win-probability forecasting. Market data is never a model input.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_NOT_IMPLEMENTED = "[yellow]not implemented yet[/yellow] (scaffold stage)"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nfl {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """NFL win-probability toolkit."""


@app.command()
def ingest(
    seasons: Annotated[
        str, typer.Option("--seasons", help="Season range, e.g. '2010-2024' or '2024'.")
    ] = "2010-2024",
    source: Annotated[
        str, typer.Option("--source", help="Source: nflverse | weather | injuries | news | all.")
    ] = "all",
) -> None:
    """Ingest raw data into ``data/raw`` and normalise it into Parquet/DuckDB."""
    console.print(f"ingest seasons={seasons} source={source}: {_NOT_IMPLEMENTED}")


@app.command()
def train(
    model: Annotated[
        str, typer.Option("--model", help="Model key: baseline | gbdt | mlp | seq | ensemble.")
    ] = "baseline",
    train_through: Annotated[
        int, typer.Option("--train-through", help="Last season included in training.")
    ] = 2022,
    val_season: Annotated[int, typer.Option("--val-season", help="Validation season.")] = 2023,
) -> None:
    """Train a model on season-based splits and write it to ``models/``."""
    console.print(
        f"train model={model} train_through={train_through} val_season={val_season}: "
        f"{_NOT_IMPLEMENTED}"
    )


@app.command()
def predict(
    game_id: Annotated[str, typer.Argument(help="Game id, e.g. '2024_18_KC_DEN'.")],
    at: Annotated[
        str | None,
        typer.Option(
            "--at",
            help="Snapshot time: ISO timestamp, 'Q4 05:00', play id, or omit for live/latest.",
        ),
    ] = None,
    model_version: Annotated[
        str | None, typer.Option("--model-version", help="Pin a model version; default is latest.")
    ] = None,
) -> None:
    """Predict win probability for a game at a point in time and log it to the ledger."""
    console.print(
        f"predict game_id={game_id} at={at or 'latest'} model_version={model_version or 'latest'}: "
        f"{_NOT_IMPLEMENTED}"
    )


@app.command()
def evaluate(
    season: Annotated[
        int | None, typer.Option("--season", help="Restrict to a single season.")
    ] = None,
    vs_market: Annotated[
        bool, typer.Option("--vs-market/--no-vs-market", help="Also report edge vs closing line.")
    ] = True,
) -> None:
    """Report log-loss, Brier score, calibration and edge vs market from the ledger."""
    console.print(f"evaluate season={season or 'all'} vs_market={vs_market}: {_NOT_IMPLEMENTED}")


if __name__ == "__main__":
    app()
