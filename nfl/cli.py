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

ingest_app = typer.Typer(help="Ingest data sources")
app.add_typer(ingest_app, name="ingest")

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


@ingest_app.command("weather")
def ingest_weather_cmd(
    seasons: Annotated[
        str, typer.Option("--seasons", help="Season range, e.g. '2010-2024' or '2024'.")
    ] = "2010-2024",
    games_parquet: Annotated[
        str | None,
        typer.Option("--games-parquet", help="Games/schedules parquet (glob allowed)."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="auto (ERA5 for past games, forecast for upcoming) | archive | forecast.",
        ),
    ] = "auto",
    force: Annotated[
        bool, typer.Option("--force/--no-force", help="Refetch seasons already on disk.")
    ] = False,
) -> None:
    """Fetch Open-Meteo weather for each game and write ``data/raw/weather/season=YYYY/``."""
    from nfl.data import weather as weather_mod

    paths = weather_mod.ingest_weather(
        seasons,
        games_parquet=games_parquet or weather_mod.DEFAULT_GAMES_PARQUET,
        mode=mode,
        force=force,
    )
    for path in paths:
        console.print(f"wrote [green]{path}[/green]")
    if not paths:
        console.print("[yellow]no seasons matched[/yellow]")


market_app = typer.Typer(help="Kalshi / closing-line market data. Evaluation only (ADR 0001).")
app.add_typer(market_app, name="market")


def _parse_seasons(seasons: str) -> list[int]:
    """``'2010-2012'`` or ``'2024'`` or ``'2022,2024'`` -> list of ints."""
    out: list[int] = []
    for chunk in seasons.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(chunk))
    return out


@ingest_app.command("closing-lines")
def ingest_closing_lines_cmd(
    seasons: Annotated[
        str, typer.Option("--seasons", help="Season range, e.g. '2010-2024' or '2024'.")
    ] = "2010-2024",
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing parquet.")] = False,
) -> None:
    """Read closing lines out of the nflverse schedule into ``Market`` rows."""
    from nfl.data.market import closing_lines

    try:
        paths = closing_lines.ingest_closing_lines(_parse_seasons(seasons), force=force)
    except closing_lines.ClosingLinesNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    for path in paths:
        console.print(f"wrote {path}")


@ingest_app.command("kalshi")
def ingest_kalshi_cmd(
    game_ids: Annotated[list[str], typer.Argument(help="Game ids, e.g. '2024_18_KC_DEN'.")],
    liquid_only: Annotated[
        bool, typer.Option("--liquid-only", help="Drop rows failing the liquidity filter.")
    ] = False,
) -> None:
    """Snapshot Kalshi NFL game markets. **Evaluation only — never a feature.**"""
    from nfl.data.market import closing_lines, kalshi

    schedule = _load_schedule(closing_lines)
    path = kalshi.ingest_kalshi(game_ids, schedule=schedule, require_liquidity=liquid_only)
    console.print(f"wrote {path}")


@ingest_app.command("sentiment")
def ingest_sentiment_cmd(
    game_ids: Annotated[list[str], typer.Argument(help="Game ids, e.g. '2024_18_KC_DEN'.")],
    hours: Annotated[
        int, typer.Option("--hours", help="Look-back window ending now, in hours.")
    ] = 168,
) -> None:
    """Score ESPN injuries + news into rule-based ``Sentiment`` rows."""
    from datetime import UTC, datetime, timedelta

    from nfl.data.sentiment import sources

    window_end = datetime.now(UTC)
    path = sources.ingest_sentiment(
        game_ids,
        window_start=window_end - timedelta(hours=hours),
        window_end=window_end,
    )
    console.print(f"wrote {path}")


def _load_schedule(closing_lines_module):  # type: ignore[no-untyped-def]
    try:
        return closing_lines_module.load_schedule()
    except closing_lines_module.ClosingLinesNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@market_app.command("snapshot")
def market_snapshot(
    game_id: Annotated[str, typer.Argument(help="Game id, e.g. '2024_18_KC_DEN'.")],
    liquid_only: Annotated[
        bool, typer.Option("--liquid-only", help="Drop the row if the book is too wide/thin.")
    ] = False,
) -> None:
    """Current Kalshi mid for a game. Evaluation/ledger only (ADR 0001)."""
    from nfl.data.market import closing_lines, kalshi

    schedule = _load_schedule(closing_lines)
    row = kalshi.snapshot_game(game_id, schedule=schedule, require_liquidity=liquid_only)
    if row is None:
        console.print(f"[yellow]no Kalshi market found for {game_id}[/yellow]")
        raise typer.Exit(1)
    p_home = "?" if row.implied_prob_home is None else f"{row.implied_prob_home:.4f}"
    console.print(f"{row.game_id} {row.kalshi_ticker} p_home={p_home}")
    console.print(
        f"  yes={row.kalshi_yes_price} no={row.kalshi_no_price} "
        f"spread={row.bid_ask_spread} oi={row.kalshi_open_interest} "
        f"collapsed={row.market_collapsed}"
    )


@market_app.command("history")
def market_history(
    game_id: Annotated[str, typer.Argument(help="Game id, e.g. '2024_18_KC_DEN'.")],
    interval: Annotated[
        int, typer.Option("--interval", help="Candle period in minutes: 1, 60 or 1440.")
    ] = 60,
    limit: Annotated[int, typer.Option("--limit", help="Max bars to print.")] = 20,
) -> None:
    """Kalshi candlestick price history for a game. Evaluation/ledger only."""
    from nfl.data.market import closing_lines, kalshi

    schedule = _load_schedule(closing_lines)
    rows = kalshi.history_game(game_id, schedule=schedule, period_interval=interval)
    if not rows:
        console.print(f"[yellow]no Kalshi history for {game_id}[/yellow]")
        raise typer.Exit(1)
    for row in rows[-limit:]:
        console.print(
            f"{row.observed_at_utc.isoformat()} p_home={row.implied_prob_home} "
            f"vol={row.kalshi_volume} oi={row.kalshi_open_interest}"
        )
    console.print(f"[dim]{len(rows)} bars ({interval}m) for {rows[0].kalshi_ticker}[/dim]")


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
