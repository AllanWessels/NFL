"""``nfl`` command-line interface.

``ingest`` is a real sub-app (``nfl ingest <source>``); ``train``, ``predict``
and ``evaluate`` are still stubs, present so the CLI surface, the ``/predict``
slash command and CI wiring can be exercised before those pipelines land.
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


ingest_app = typer.Typer(help="Ingest data sources")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("nflverse")
def ingest_nflverse_cmd(
    seasons: Annotated[
        str, typer.Option("--seasons", help="Season range, e.g. '2010-2025' or '2024'.")
    ] = "2010-2025",
    tables: Annotated[
        str | None,
        typer.Option("--tables", help="Comma-separated subset of tables; default is all."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download and overwrite existing partitions.")
    ] = False,
) -> None:
    """Ingest nflverse play-by-play, schedules and weekly tables into ``data/raw``."""
    from nfl.data.coach_records import ingest_coach_records
    from nfl.data.nflverse import TABLE_NAMES, IngestResult, ingest_nflverse, parse_seasons
    from nfl.data.store import open_db, refresh_views

    try:
        season_list = parse_seasons(seasons)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--seasons") from exc

    table_list = [t.strip() for t in tables.split(",") if t.strip()] if tables else None
    if table_list:
        unknown = [t for t in table_list if t not in TABLE_NAMES]
        if unknown:
            raise typer.BadParameter(
                f"unknown table(s) {unknown}; known: {list(TABLE_NAMES)}", param_hint="--tables"
            )

    def report(result: IngestResult) -> None:
        if result.written:
            console.print(
                f"  [green]wrote[/green] {result.table} {result.season}: {result.rows:,} rows"
            )
        elif result.reason not in {"exists", ""}:
            console.print(f"  [dim]skip[/dim]  {result.table} {result.season}: {result.reason}")

    console.print(f"ingest nflverse seasons={season_list[0]}-{season_list[-1]} force={force}")
    results: list[IngestResult] = []
    root = ingest_nflverse(season_list, table_list, force=force, results=results, on_result=report)

    if table_list is None or "schedules" in table_list:
        built: list[int] = []
        ingest_coach_records(season_list, force=force, written=built)
        if built:
            console.print(f"  [green]built[/green] coach_records for {len(built)} season(s)")

    con = open_db()
    try:
        views = refresh_views(con)
    finally:
        con.close()
    written = sum(r.rows for r in results if r.written)
    console.print(f"{root}: {written:,} rows written, {len(views)} views registered")


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
