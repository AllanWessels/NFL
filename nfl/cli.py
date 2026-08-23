"""``nfl`` command-line interface.

``ingest``, ``features``, ``market``, ``live``, ``train``, ``predict``,
``evaluate`` and ``ledger`` all run real code; ``predict`` is the end-to-end
inference path behind the ``/predict`` slash command.

Every command imports its implementation *inside the function body*. That is
deliberate: ``nfl --help`` must stay instant, and the TensorFlow models must not
be imported by someone running ``nfl ingest``.
"""

from __future__ import annotations

from typing import Annotated, Any

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


#: ``--model`` key -> ``(module, function)``. Imported lazily at dispatch time:
#: ``nfl --help`` must not import TensorFlow, and a model that has not landed
#: yet should produce a sentence rather than a traceback.
TRAINERS: dict[str, tuple[str, str]] = {
    "naive": ("nfl.models.naive", "train_naive"),
    "xgb": ("nfl.models.gbdt", "train_xgboost"),
    "mlp": ("nfl.models.tf_mlp", "train_mlp"),
    "seq": ("nfl.models.tf_seq", "train_seq"),
    "ensemble": ("nfl.models.train_ensemble", "train_ensemble"),
}

#: Older spellings kept working so existing scripts and docs do not break.
MODEL_ALIASES: dict[str, str] = {
    "baseline": "naive",
    "gbdt": "xgb",
    "tf_mlp": "mlp",
    "tf_seq": "seq",
}


def _resolve_model_key(model: str) -> str:
    key = MODEL_ALIASES.get(model, model)
    if key not in TRAINERS:
        raise typer.BadParameter(
            f"unknown model {model!r}; expected one of {sorted(TRAINERS)} "
            f"(aliases: {sorted(MODEL_ALIASES)})",
            param_hint="--model",
        )
    return key


def _load_trainer(key: str) -> Any:
    """Import the sibling trainer, or exit saying what is missing and where."""
    import importlib

    module_name, func_name = TRAINERS[key]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        path = module_name.replace(".", "/")
        console.print(
            f"[red]model '{key}' is not available[/red]: cannot import {module_name} ({exc}).\n"
            f"  It lives in {path}.py, which is not in this checkout yet."
        )
        raise typer.Exit(1) from exc
    func = getattr(module, func_name, None)
    if func is None:
        console.print(
            f"[red]{module_name} has no {func_name}()[/red]; the trainer entrypoint for "
            f"'{key}' is expected to be `def {func_name}(seasons=..., ...)`."
        )
        raise typer.Exit(1)
    return func


@app.command()
def train(
    model: Annotated[
        str,
        typer.Option("--model", help="Model key: naive | xgb | mlp | seq | ensemble."),
    ] = "naive",
    seasons: Annotated[
        str | None,
        typer.Option("--seasons", help="Seasons to train on, e.g. '2010-2022' or '2018,2020'."),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help=(
                "Season split: 'train_through/val_season/test_from', 'rolling:<season>', "
                "or a bare season (= rolling on it). Default is the rolling window "
                "dated from the schedules on disk: train <= L-2, val L-1, test L."
            ),
        ),
    ] = None,
    members: Annotated[
        str | None,
        typer.Option(
            "--members",
            help=(
                "Ensemble only: comma-separated 'name[:version]'; the version pins a "
                "registry artifact. Default is the ADR 0003 three, each at its latest."
            ),
        ),
    ] = None,
    meta: Annotated[
        str | None,
        typer.Option("--meta", help="Ensemble only: logit_mean | mean | logistic_stacker."),
    ] = None,
) -> None:
    """Train a model on season-based splits and write it to ``data/models/``.

    Each model lives in its own module and is imported only when selected, so
    the GBDT path never pays for TensorFlow's import and vice versa.

    The split rolls with the data — train <= L-2, val L-1, test L, where L is
    the newest season whose regular season has finished on disk — and whichever
    split is used is written into the artifact's manifest, because the
    promotion gate refuses to compare two models validated on different seasons.
    ``--split`` pins it.

    ``--model ensemble`` does not train anything itself: it loads the latest
    registry artifact of each member, fits the meta-learner and the phase-aware
    calibrator on val, and writes the blended model back to the registry.
    """
    import inspect

    from nfl.features.splits import resolve_split

    key = _resolve_model_key(model)
    trainer = _load_trainer(key)

    try:
        resolved_split = resolve_split(split)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--split") from exc

    season_list = _parse_seasons(seasons) if seasons else None
    candidates: dict[str, Any] = {
        "seasons": season_list,
        "split": resolved_split,
        "members": [m.strip() for m in members.split(",") if m.strip()] if members else None,
        "meta": meta,
    }
    # Sibling trainers are plain functions with varying signatures; pass only
    # what each one actually accepts rather than assuming a common shape.
    try:
        params = inspect.signature(trainer).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        params = {}  # type: ignore[assignment]
    kwargs = {k: v for k, v in candidates.items() if k in params and v is not None}

    scope = seasons if seasons else "all on disk"
    console.print(f"train model={key} seasons={scope} split={resolved_split.label}")
    _render_fit_result(trainer(**kwargs), key)


def _render_fit_result(result: object, key: str) -> None:
    """Print whatever the trainer returned; a ``FitResult`` gets a metrics table."""
    from rich.table import Table

    # Trainers return either a bare ``FitResult`` or ``(model, FitResult)``.
    if isinstance(result, tuple) and len(result) == 2 and hasattr(result[1], "metrics"):
        result = result[1]
    name = getattr(result, "model_name", key)
    version = getattr(result, "version", None)
    if version is None:
        console.print(f"[green]trained[/green] {name}: {result}")
        return
    console.print(f"[green]trained[/green] {name} version={version}")
    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, dict) and metrics:
        table = Table(title="metrics")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for metric, value in metrics.items():
            table.add_row(str(metric), f"{value:.6f}" if isinstance(value, float) else str(value))
        console.print(table)


@app.command()
def promote(
    candidate: Annotated[
        str, typer.Option("--candidate", help="Model name to promote, e.g. 'ensemble_v1'.")
    ],
    version: Annotated[
        str | None, typer.Option("--version", help="Candidate version; default is latest.")
    ] = None,
    floor: Annotated[
        str, typer.Option("--floor", help="Model the candidate must beat (ADR 0003 §5).")
    ] = "naive_score_time",
    floor_version: Annotated[
        str | None, typer.Option("--floor-version", help="Floor version; default is latest.")
    ] = None,
    stage: Annotated[str, typer.Option("--stage", help="Stage pointer to move.")] = "production",
    parity_samples: Annotated[
        int | None,
        typer.Option(
            "--parity-samples",
            help=(
                "Val snapshots scored down both the serving and the offline path "
                "(default 20). 0 skips the serving-parity condition, which makes the "
                "rest of the table a comparison of artifacts rather than a promotion "
                "decision."
            ),
        ),
    ] = None,
    parity_tolerance: Annotated[
        float | None,
        typer.Option(
            "--parity-tolerance",
            help="Largest |p_serving - p_offline| allowed (default 0.02).",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Promote even if the gate fails. Says so, loudly.")
    ] = False,
) -> None:
    """Promote a model to PRODUCTION if it clears the ADR 0003 §5 gate.

    The candidate must beat the floor on val Q4@600/300/120 for **both**
    log-loss and Brier, be no worse on the not-collapsed subset, and **serve
    what it scored**: ``--parity-samples`` snapshots are run through
    ``nfl predict`` and compared against the offline dataset row for the same
    play. The comparison table prints either way — a refused promotion should
    say exactly which number refused it.
    """
    from rich.table import Table

    from nfl.models import registry
    from nfl.models.train_ensemble import (
        SERVING_PARITY_SAMPLES,
        SERVING_PARITY_TOLERANCE,
        gate_table,
        promotion_gate,
    )

    try:
        result = promotion_gate(
            candidate,
            version,
            floor=floor,
            floor_version=floor_version,
            root=None,
            parity_samples=SERVING_PARITY_SAMPLES if parity_samples is None else parity_samples,
            parity_tolerance=(
                SERVING_PARITY_TOLERANCE if parity_tolerance is None else parity_tolerance
            ),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"[red]cannot evaluate the promotion gate[/red]: {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"[bold]promotion gate[/bold] candidate="
        f"[cyan]{result.candidate}[/cyan]/[dim]{result.candidate_version}[/dim] "
        f"floor=[cyan]{result.floor}[/cyan]/[dim]{result.floor_version}[/dim]"
    )
    for note in result.notes:
        console.print(f"  [yellow]{note}[/yellow]")

    table = Table(title="val slices — lower is better")
    for header in ("slice", "metric", "candidate", "floor", "delta", "rule", "verdict"):
        table.add_column(
            header, justify="right" if header in {"candidate", "floor", "delta"} else "left"
        )
    for row in gate_table(result).iter_rows(named=True):
        verdict = row["verdict"]
        table.add_row(
            row["slice"],
            row["metric"],
            f"{row['candidate']:.6f}",
            f"{row['floor']:.6f}",
            f"{row['delta']:+.6f}",
            row["rule"],
            f"[green]{verdict}[/green]" if verdict == "pass" else f"[red]{verdict}[/red]",
        )
    console.print(table)

    if result.serving_parity is not None and result.serving_parity.samples:
        parity = result.serving_parity
        detail = Table(
            title=(
                f"serving parity — |p_serving - p_offline|, tolerance {parity.tolerance:g} "
                f"(worst {parity.worst:.4f}, mean {parity.mean:.4f})"
            )
        )
        for header in ("game", "at", "play", "serving", "offline", "delta", "verdict"):
            detail.add_column(
                header, justify="right" if header not in {"game", "at", "verdict"} else "left"
            )
        for row in parity.table().iter_rows(named=True):
            verdict = row["verdict"]
            detail.add_row(
                row["game_id"],
                row["at"],
                str(row["play_index"]),
                f"{row['p_serving']:.4f}",
                f"{row['p_offline']:.4f}",
                f"{row['delta']:.4f}",
                f"[green]{verdict}[/green]" if verdict == "pass" else f"[red]{verdict}[/red]",
            )
        console.print(detail)

    if not result.passed and not force:
        console.print(
            f"[red]NOT promoted[/red]: {result.candidate}/{result.candidate_version} does not "
            f"clear the floor. Retrain, or re-run with --force if you know why."
        )
        raise typer.Exit(1)

    if not result.passed:
        console.print(
            "[bold red]!! FORCED PROMOTION !!"
            "[/bold red] the gate FAILED and --force overrode it.\n"
            "[bold red]   This model is going to PRODUCTION worse than the naive floor "
            "on the slices the project is judged on (ADR 0003 §5).[/bold red]"
        )

    pointer = registry.promote(result.candidate, result.candidate_version, stage)
    console.print(
        f"[green]promoted[/green] {result.candidate}/{result.candidate_version} "
        f"to {stage.upper()} -> {pointer}"
    )


@app.command()
def retrain(
    season: Annotated[
        int | None,
        typer.Option("--season", help="Season to ingest; default is the one in progress."),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help=(
                "Season split for every fit: 'train_through/val_season/test_from', "
                "'rolling:<season>', or a bare season. Default is the rolling window."
            ),
        ),
    ] = None,
    skip_ingest: Annotated[
        bool, typer.Option("--skip-ingest", help="Do not touch the network; use what is on disk.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan — season, split, step order — only.")
    ] = False,
    parity_samples: Annotated[
        int | None,
        typer.Option("--parity-samples", help="Serving-parity samples in the gate (default 20)."),
    ] = None,
) -> None:
    """Ingest, rebuild, retrain, gate, promote, resolve and evaluate — in that order.

    The weekly loop as one command. Every step is timed, the whole run is
    written to ``data/reports/retrain-<timestamp>.md``, and the summary at the
    end answers the three questions the run exists to answer: did the gate pass,
    what is PRODUCTION pointing at now, and has the served model drifted.

    A step that raises stops the chain. A gate that *refuses* does not — it is
    the gate working, and the ledger and drift readings that follow describe the
    model still in production, which is exactly what you want to see when a
    candidate has just failed to replace it. The command exits non-zero either
    way.
    """
    from nfl import retrain as retrain_mod

    try:
        result = retrain_mod.run_retrain(
            season=season,
            split=split,
            skip_ingest=skip_ingest,
            dry_run=dry_run,
            echo=console.print,
            parity_samples=parity_samples,
        )
    except ValueError as exc:
        console.print(f"[red]cannot start the retrain[/red]: {exc}")
        raise typer.Exit(1) from exc

    console.print()
    retrain_mod.render_summary(result, console)
    if result.report_path is not None:
        console.print(f"\nrun log: [green]{result.report_path}[/green]")

    failed = result.failed_step
    if failed is not None:
        console.print(
            f"\n[red]retrain stopped at '{failed.name}'[/red]: {failed.detail}\n"
            "Nothing downstream ran, because it would have been reading stale inputs."
        )
        raise typer.Exit(1)
    if result.gate is not None and not result.gate.passed:
        console.print(
            f"\n[yellow]NOT promoted[/yellow]: {result.candidate}/{result.candidate_version} "
            "did not clear the ADR 0003 §5 gate. PRODUCTION is unchanged. "
            "`nfl promote --candidate ... --force` exists, and should stay unused."
        )
        raise typer.Exit(1)


@app.command()
def predict(
    game: Annotated[
        str,
        typer.Argument(
            help=(
                "Game id ('2024_18_KC_DEN'), matchup ('DAL@PHI', 'PHI vs DAL'), "
                "one team ('PHI', 'Eagles'), a Kalshi ticker, or an ESPN event id."
            )
        ),
    ],
    at: Annotated[
        str | None,
        typer.Option("--at", help="Rewind to a game time, e.g. 'Q4 05:00'; omit for latest."),
    ] = None,
    season: Annotated[
        int | None,
        typer.Option("--season", help="Disambiguate a matchup or team to one season."),
    ] = None,
    week: Annotated[
        int | None, typer.Option("--week", help="Disambiguate a matchup or team to one week.")
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Feed: auto (replay a finished game, else ESPN) | espn | replay.",
        ),
    ] = "auto",
    model: Annotated[
        str,
        typer.Option("--model", help="'production', a model name, or 'name:version'."),
    ] = "production",
    no_ledger: Annotated[
        bool, typer.Option("--no-ledger", help="Do not append the prediction to the ledger.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the result as JSON instead of a panel.")
    ] = False,
) -> None:
    """Predict win probability for a game at a point in time and log it to the ledger.

    GAME accepts anything :func:`nfl.inference.resolve.resolve_game` understands;
    the resolution is printed before the panel so the answer always says which
    game it is about. ``--at`` turns off the preference for an upcoming game:
    asking for a past moment means nearest-kickoff wins outright.
    """
    import json as json_mod
    from typing import cast

    from nfl.inference.predict import PredictionError, Source, predict_game, render_prediction
    from nfl.inference.resolve import GameSpecError, resolve_game

    if source not in ("auto", "espn", "replay"):
        raise typer.BadParameter(
            f"unknown source {source!r}; expected auto | espn | replay", param_hint="--source"
        )
    try:
        ref = resolve_game(game, season=season, week=week, prefer_upcoming=at is None)
    except GameSpecError as exc:
        console.print(f"[red]cannot resolve {game!r}[/red]\n{exc}")
        raise typer.Exit(1) from exc
    except (FileNotFoundError, KeyError) as exc:
        console.print(f"[red]cannot read the schedules table[/red]: {exc}")
        raise typer.Exit(1) from exc

    if not as_json:
        console.print(f"[dim]{ref.describe()}[/dim]")
        if ref.note:
            console.print(f"[yellow]{ref.note}[/yellow]")

    try:
        result = predict_game(
            ref.game_id,
            at=at,
            source=cast("Source", source),
            model=model,
            write_ledger=not no_ledger,
        )
    except PredictionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(f"[red]cannot predict {ref.game_id}[/red]: {exc}")
        raise typer.Exit(1) from exc

    if as_json:
        console.print_json(json_mod.dumps(result.to_dict()))
        return
    render_prediction(result, console)


@app.command()
def evaluate(
    model: Annotated[
        str | None, typer.Option("--model", help="Restrict to one model name.")
    ] = None,
    version: Annotated[
        str | None, typer.Option("--version", help="Restrict to one model version.")
    ] = None,
    season: Annotated[
        int | None, typer.Option("--season", help="Restrict to a single season.")
    ] = None,
    ledger: Annotated[
        bool,
        typer.Option(
            "--ledger/--backtest",
            help="Score logged predictions (default) or run an offline backtest.",
        ),
    ] = True,
    vs_market: Annotated[
        bool, typer.Option("--vs-market/--no-vs-market", help="Also report edge vs closing line.")
    ] = True,
    threshold: Annotated[
        float, typer.Option("--threshold", help="Minimum |p - market| before the paper bet.")
    ] = 0.05,
    fee_coefficient: Annotated[
        float, typer.Option("--fee-coefficient", help="Kalshi fee coefficient (see evaluate.py).")
    ] = 0.07,
    two_sided_fees: Annotated[
        bool,
        typer.Option(
            "--two-sided-fees/--one-sided-fees",
            help="Charge the trading fee on entry and exit, or only on entry.",
        ),
    ] = True,
    baseline_log_loss: Annotated[
        float | None,
        typer.Option(
            "--baseline-log-loss",
            help=(
                "Training-time log loss for the drift trigger. Omit it and the number "
                "is read from each model's own manifest."
            ),
        ),
    ] = None,
    window: Annotated[
        int, typer.Option("--window", help="Rolling window size for the drift trigger.")
    ] = 500,
    drift_threshold: Annotated[
        float, typer.Option("--drift-threshold", help="Absolute log-loss drift that flags retrain.")
    ] = 0.02,
    markdown: Annotated[
        bool, typer.Option("--markdown/--no-markdown", help="Also write data/reports/<name>.md.")
    ] = False,
) -> None:
    """Report log-loss, Brier, calibration, edge vs market and the drift flag.

    Reads the prediction ledger by default. ``--backtest`` instead loads the
    model from the registry and replays it over the assembled dataset — the two
    paths produce the same frame schema, so the numbers are comparable.

    The **retrain trigger** prints for every model version: the last
    ``--window`` resolved predictions are scored and compared against what that
    artifact's manifest says it scored at training time, and a drift of more
    than ``--drift-threshold`` nats flags ``RETRAIN``.
    """
    from nfl.logging import evaluate as evaluate_mod

    seasons = [season] if season is not None else None
    kwargs: dict[str, Any] = {
        "threshold": threshold,
        "fee_coefficient": fee_coefficient,
        "two_sided_fees": two_sided_fees,
        "with_market": vs_market,
        "window": window,
        "drift_threshold": drift_threshold,
        "baselines": ({model or version or "": baseline_log_loss} if baseline_log_loss else None),
    }

    if ledger:
        report = evaluate_mod.evaluate_ledger(
            seasons=seasons, model_name=model, model_version=version, **kwargs
        )
    else:
        frame = _backtest_frame(model, version, seasons)
        report = evaluate_mod.build_report(frame, source="backtest", **kwargs)

    evaluate_mod.render_report(report, console)
    if markdown:
        path = evaluate_mod.write_markdown(report)
        console.print(f"\nwrote [green]{path}[/green]")


def _backtest_frame(model: str | None, version: str | None, seasons: list[int] | None) -> Any:
    """Load a trained model from the registry and replay it over the dataset."""
    if not model:
        raise typer.BadParameter("--backtest needs --model NAME", param_hint="--model")
    import importlib

    try:
        # ``importlib`` rather than a plain import: the module is written by
        # another agent and may not exist here, and a static import would make
        # that a type-check error instead of a runtime message.
        registry = importlib.import_module("nfl.models.registry")
    except ImportError as exc:
        console.print(
            f"[red]--backtest needs nfl/models/registry.py[/red] ({exc}); "
            "it is not in this checkout yet. Use --ledger."
        )
        raise typer.Exit(1) from exc

    from nfl.features.assemble import read_dataset
    from nfl.logging.backtest import backtest

    try:
        resolved_version = version or registry.latest(model)
        loaded = registry.load(model, resolved_version)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        console.print(
            f"[red]no trained artifact for model '{model}' in the registry[/red] ({exc}); "
            "run `nfl train` first."
        )
        raise typer.Exit(1) from exc
    frame = read_dataset(seasons)
    return backtest(loaded, frame, model_name=model, model_version=resolved_version)


ledger_app = typer.Typer(help="The prediction ledger (append-only parquet + DuckDB view).")
app.add_typer(ledger_app, name="ledger")


def _schedules_with_scores() -> Any:
    """The nflverse schedule frame, final scores included, for ledger resolution."""
    import polars as pl

    from nfl.data.market import closing_lines
    from nfl.features.assemble import load_raw_table

    try:
        return load_raw_table("schedules")
    except FileNotFoundError:
        pass
    path = closing_lines.find_schedule_file()
    if path is None:
        console.print(
            "[red]no nflverse schedule on disk[/red]; run "
            "`nfl ingest nflverse --tables schedules` first"
        )
        raise typer.Exit(1)
    return pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)


@ledger_app.command("resolve")
def ledger_resolve() -> None:
    """Fill outcome columns for games that have finished. Idempotent."""
    from nfl.logging import ledger as ledger_mod

    schedules = _schedules_with_scores()
    try:
        report = ledger_mod.resolve_ledger(schedules)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"ledger resolve: {report.rows:,} row(s) scanned, "
        f"[green]{report.newly_resolved:,}[/green] newly resolved, "
        f"{report.still_open:,} still open ({len(report.files)} file(s) rewritten)"
    )


@ledger_app.command("show")
def ledger_show(
    last: Annotated[int, typer.Option("--last", "-n", help="How many rows to show.")] = 20,
    model: Annotated[str | None, typer.Option("--model", help="Filter by model name.")] = None,
    game_id: Annotated[str | None, typer.Option("--game-id", help="Filter by game.")] = None,
) -> None:
    """Print the most recent ledger rows."""
    from rich.table import Table

    from nfl.logging import ledger as ledger_mod

    frame = ledger_mod.read_ledger(model_name=model, game_ids=[game_id] if game_id else None)
    if frame.is_empty():
        console.print("[yellow]ledger is empty[/yellow]")
        return

    table = Table(title=f"prediction ledger (last {min(last, frame.height)} of {frame.height:,})")
    for header in ("predicted_at", "game_id", "clock", "model", "version", "p_home", "conf", "won"):
        table.add_column(header, justify="right" if header in {"p_home", "conf"} else "left")
    for row in frame.tail(last).iter_rows(named=True):
        stamp = row["predicted_at_utc"]
        won = row["resolved_home_win"]
        table.add_row(
            f"{stamp:%Y-%m-%d %H:%M}" if stamp else "—",
            str(row["game_id"]),
            "—"
            if row["snapshot_game_seconds_remaining"] is None
            else f"{row['snapshot_game_seconds_remaining']}s",
            str(row["model_name"] or "—"),
            str(row["model_version"] or "—"),
            f"{row['p_home']:.4f}",
            "—" if row["confidence"] is None else f"{row['confidence']:.3f}",
            "—" if won is None else ("home" if won else "away"),
        )
    console.print(table)
    resolved = int(frame.get_column("resolved_home_win").is_not_null().sum())
    console.print(f"[dim]{resolved:,} of {frame.height:,} resolved[/dim]")


@ledger_app.command("register")
def ledger_register() -> None:
    """(Re)create the ``prediction_ledger`` DuckDB view over the parquet lake."""
    from nfl.data.store import open_db
    from nfl.logging import ledger as ledger_mod

    con = open_db()
    try:
        views = ledger_mod.register_ledger_views(con)
    finally:
        con.close()
    if views:
        console.print(f"registered view [green]{views[0]}[/green]")
    else:
        console.print("[yellow]ledger is empty; no view registered[/yellow]")


features_app = typer.Typer(help="Build feature tables")
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    seasons: Annotated[
        str, typer.Option("--seasons", help="Season range, e.g. '2010-2024' or '2024'.")
    ] = "2010-2024",
    out: Annotated[
        str | None,
        typer.Option("--out", help="Root to write under; default is data/processed."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Assemble and report without writing parquet.")
    ] = False,
) -> None:
    """Assemble every available feature builder into ``data/processed/dataset``."""
    from pathlib import Path

    from nfl.data.nflverse import parse_seasons
    from nfl.features.assemble import AssemblyReport, assemble_dataset, feature_columns

    try:
        season_list = parse_seasons(seasons)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--seasons") from exc

    report = AssemblyReport()
    console.print(f"features build seasons={season_list[0]}-{season_list[-1]}")
    df = assemble_dataset(
        season_list,
        write=not dry_run,
        root=Path(out) if out else None,
        report=report,
    )
    for name, reason in report.skipped:
        console.print(f"  [dim]skip[/dim]  {name}: {reason}")
    for name in report.ran:
        console.print(f"  [green]built[/green] {name}")
    games = df.get_column("meta__game_id").n_unique()
    console.print(
        f"{df.height:,} rows / {games:,} games / {len(feature_columns(df)):,} features"
        + (" [yellow](dry run, nothing written)[/yellow]" if dry_run else "")
    )


@features_app.command("inspect")
def features_inspect(
    seasons: Annotated[
        str | None, typer.Option("--seasons", help="Restrict to seasons, e.g. '2023-2024'.")
    ] = None,
    top: Annotated[int, typer.Option("--top", help="How many features to list.")] = 15,
    path: Annotated[
        str | None, typer.Option("--path", help="Dataset root; default is data/processed.")
    ] = None,
) -> None:
    """Row/null/variance summary of the assembled dataset."""
    from pathlib import Path

    import polars as pl
    from rich.table import Table

    from nfl.features.assemble import LABEL_COLUMN, feature_columns, read_dataset

    try:
        df = read_dataset(seasons, root=Path(path) if path else None)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    feats = feature_columns(df)
    seasons_present = sorted(
        s for s in df.get_column("meta__season").unique().to_list() if s is not None
    )
    labelled = df.get_column(LABEL_COLUMN).drop_nulls()
    console.print(
        f"{df.height:,} rows / {df.get_column('meta__game_id').n_unique():,} games / "
        f"{len(feats):,} features / seasons {seasons_present[0]}-{seasons_present[-1]}"
    )
    if labelled.len():
        win_rate = labelled.cast(pl.Float64).sum() / labelled.len()
        console.print(
            f"label {LABEL_COLUMN}: {labelled.len():,} labelled rows, home win rate {win_rate:.4f}"
        )
    else:
        console.print(f"label {LABEL_COLUMN}: [yellow]no labelled rows[/yellow]")
    if not feats:
        console.print("[yellow]no feature columns — are the builders installed?[/yellow]")
        return

    nulls = df.select([pl.col(c).null_count().alias(c) for c in feats]).row(0)
    variances = df.select(
        [pl.col(c).cast(pl.Float64, strict=False).var().alias(c) for c in feats]
    ).row(0)
    stats = [
        (c, (n or 0) / max(df.height, 1), v if v is not None else float("nan"))
        for c, n, v in zip(feats, nulls, variances, strict=True)
    ]

    variance_table = Table(title=f"top {top} features by variance")
    variance_table.add_column("feature")
    variance_table.add_column("variance", justify="right")
    variance_table.add_column("null rate", justify="right")
    for name, null_rate, var in sorted(
        stats, key=lambda s: s[2] if s[2] == s[2] else -1.0, reverse=True
    )[:top]:
        variance_table.add_row(name, f"{var:,.3f}", f"{null_rate:.1%}")
    console.print(variance_table)

    null_table = Table(title=f"top {top} features by null rate")
    null_table.add_column("feature")
    null_table.add_column("null rate", justify="right")
    for name, null_rate, _ in sorted(stats, key=lambda s: s[1], reverse=True)[:top]:
        if null_rate == 0:
            break
        null_table.add_row(name, f"{null_rate:.1%}")
    if null_table.row_count:
        console.print(null_table)
    else:
        console.print("[green]no nulls in any feature column[/green]")


live_app = typer.Typer(help="Live and replayed in-game feeds.", no_args_is_help=True)
app.add_typer(live_app, name="live")


@live_app.command("snapshot")
def live_snapshot(
    game_id: Annotated[
        str, typer.Argument(help="nflverse game id ('2024_01_BAL_KC') or ESPN event id.")
    ],
    at: Annotated[
        str | None,
        typer.Option("--at", help="Rewind to a game time, e.g. 'Q4 05:00' or 'OT 10:00'."),
    ] = None,
    source: Annotated[
        str, typer.Option("--source", help="Feed: espn (live wire) | replay (nflverse pbp).")
    ] = "espn",
    schedules: Annotated[
        str | None,
        typer.Option("--schedules", help="Override the schedules parquet path or directory."),
    ] = None,
    pbp: Annotated[
        str | None, typer.Option("--pbp", help="Override the play-by-play parquet path (replay).")
    ] = None,
) -> None:
    """Print the current (or rewound) state of a game from a live feed."""
    from nfl.data.live.render import render_snapshot

    if source == "espn":
        from nfl.data.live.espn import EspnLiveFeed

        with EspnLiveFeed(schedules=schedules) as feed:
            snapshot = feed.snapshot(game_id, at=at)
    elif source == "replay":
        from nfl.data.live.replay import HistoricalReplayFeed

        feed_ = HistoricalReplayFeed(pbp=pbp, schedules=schedules)
        snapshot = feed_.snapshot(game_id, at=at)
    else:
        console.print(f"[red]unknown --source {source!r}[/red] (expected 'espn' or 'replay')")
        raise typer.Exit(code=2)

    render_snapshot(snapshot, console)


if __name__ == "__main__":
    app()
