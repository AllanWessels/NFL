"""``coach_records`` — as-of-date head-coach records, one row per game.

nflverse has no coaches table. It does not need one: ``schedules`` carries
``home_coach``/``away_coach`` per *game*, which handles mid-season firings and
interim promotions correctly for free. This module turns that column pair into
the record fields ``docs/DATA_MODEL.md`` §4 asks for.

**The whole point of this module is the as-of rule.** ``docs/DATA_MODEL.md``
marks ``career_wins``/``career_losses`` "**Must exclude the current game.**"
A coach's record is trivially computable *after* the season; computed that way
it is a direct function of the outcomes we are trying to predict, and a model
fed it would score beautifully and be worthless. So every count here is over
games that had **kicked off strictly before** the game on the row:

* the current game never contributes to its own row;
* games later in the season never contribute;
* games with no result yet (scheduled, not played) contribute nothing to
  anyone's record.

Implemented as a shift-then-cumulative-sum over a coach-major ordering, which
makes the exclusion structural rather than a filter someone can forget.
``tests/test_coach_records.py`` pins it both ways: directly, and by the
recompute-with-the-future-deleted invariant from the leakage taxonomy.

Output: ``data/processed/coach_records/season=YYYY/part-0.parquet``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from nfl.config import PROCESSED_DIR
from nfl.data.nflverse import SCHEMA_VERSION, ingested_at_now

__all__ = [
    "SOURCE",
    "TABLE",
    "build_coach_records",
    "coach_records_path",
    "ingest_coach_records",
    "load_schedules_frame",
]

TABLE = "coach_records"
SOURCE = "derived"

#: Columns of the schedules frame this builder needs.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_coach",
    "away_coach",
    "home_score",
    "away_score",
)

_SIDES = ("home", "away")


def coach_records_path(season: int, root: Path | None = None) -> Path:
    """``<processed>/coach_records/season=YYYY/part-0.parquet``."""
    root = PROCESSED_DIR if root is None else Path(root)
    return root / TABLE / f"season={season}" / "part-0.parquet"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_coach_records(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per game with as-of-date records for both head coaches.

    ``schedules`` must be the nflverse games frame (or a subset of it carrying
    :data:`REQUIRED_COLUMNS`). Pass **every** season you have: career records
    are cumulative across seasons, so slicing the input to one season silently
    produces career counts that restart at zero.

    Returned columns, for ``side`` in ``{home, away}``:

    ``{side}_coach``
        The coach's name, straight from schedules.
    ``{side}_coach_career_{wins,losses,ties,games}``
        Over every completed game that coach has coached, any team, any season,
        before this one.
    ``{side}_coach_career_win_pct``
        ``(wins + 0.5 * ties) / games``; ``0.0`` for a coach's first game.
    ``{side}_coach_season_{wins,losses,ties,games}``
        The same, restricted to the current season.

    Career and season counts both include postseason games; ``game_type`` is
    kept on the row so a consumer can split them if it wants to.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in schedules.columns]
    if missing:
        raise ValueError(f"schedules frame is missing required columns: {missing}")

    games = _normalise(schedules)
    long = _to_coach_rows(games)
    long = _accumulate(long)

    out = games.select(_carry_columns(games))
    for side in _SIDES:
        side_frame = long.filter(pl.col("side") == side).select(
            "game_id",
            pl.col("coach").alias(f"{side}_coach"),
            *[
                pl.col(metric).alias(f"{side}_coach_{metric}")
                for metric in (
                    "career_wins",
                    "career_losses",
                    "career_ties",
                    "career_games",
                    "season_wins",
                    "season_losses",
                    "season_ties",
                    "season_games",
                )
            ],
        )
        out = out.join(side_frame, on="game_id", how="left")

    out = out.with_columns(
        [_win_pct(f"{side}_coach_career") for side in _SIDES]
        + [_win_pct(f"{side}_coach_season") for side in _SIDES]
    )
    return out.sort("kickoff", "game_id").with_columns(
        pl.lit(SOURCE, dtype=pl.String).alias("source"),
        pl.lit(ingested_at_now()).alias("ingested_at"),
        pl.lit(SCHEMA_VERSION, dtype=pl.String).alias("schema_version"),
    )


def _win_pct(prefix: str) -> pl.Expr:
    games = pl.col(f"{prefix}_games")
    wins = pl.col(f"{prefix}_wins")
    ties = pl.col(f"{prefix}_ties")
    return (
        pl.when(games > 0)
        .then((wins + 0.5 * ties) / games)
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias(f"{prefix}_win_pct")
    )


def _carry_columns(games: pl.DataFrame) -> list[str]:
    wanted = ["game_id", "season", "week", "game_type", "gameday", "kickoff", "home_team", "away_team"]
    return [c for c in wanted if c in games.columns]


def _normalise(schedules: pl.DataFrame) -> pl.DataFrame:
    """Add the ordering key and the per-side outcome flags."""
    df = schedules.with_columns(
        pl.col("gameday").cast(pl.String).str.to_date("%Y-%m-%d", strict=False).alias("game_date"),
    )
    # Order by date, then by kickoff time when we have it: a coach never has two
    # games on one day, so this only has to be stable, not physically exact.
    time_key = (
        pl.col("gametime").cast(pl.String).fill_null("00:00")
        if "gametime" in df.columns
        else pl.lit("00:00")
    )
    df = df.with_columns(
        pl.concat_str([pl.col("game_date").cast(pl.String).fill_null("9999-12-31"), time_key], separator=" ")
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False)
        .alias("kickoff")
    )
    played = pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    margin = pl.col("home_score") - pl.col("away_score")
    return df.with_columns(
        played.alias("played"),
        (played & (margin > 0)).cast(pl.Int64).alias("home_won"),
        (played & (margin < 0)).cast(pl.Int64).alias("away_won"),
        (played & (margin == 0)).cast(pl.Int64).alias("tied"),
    )


def _to_coach_rows(games: pl.DataFrame) -> pl.DataFrame:
    """Explode each game into two coach-perspective rows."""
    frames = []
    for side in _SIDES:
        other = "away" if side == "home" else "home"
        frames.append(
            games.select(
                pl.col("game_id"),
                pl.col("season"),
                pl.col("kickoff"),
                pl.lit(side, dtype=pl.String).alias("side"),
                pl.col(f"{side}_coach").alias("coach"),
                pl.col(f"{side}_won").fill_null(0).alias("won"),
                pl.col(f"{other}_won").fill_null(0).alias("lost"),
                pl.col("tied").fill_null(0).alias("tied"),
            )
        )
    return pl.concat(frames, how="vertical")


def _accumulate(long: pl.DataFrame) -> pl.DataFrame:
    """Shift-then-cumsum: every count covers strictly earlier games only.

    ``cum_sum().shift(1)`` over a coach's games ordered by kickoff is what makes
    the as-of guarantee structural — the row's own outcome is shifted out before
    it can ever be added in.
    """
    ordered = long.sort("kickoff", "game_id", "side")
    metrics = {"wins": "won", "losses": "lost", "ties": "tied"}

    career = [
        pl.col(src).cum_sum().shift(1).fill_null(0).over("coach", order_by=["kickoff", "game_id"]).alias(f"career_{name}")
        for name, src in metrics.items()
    ]
    season = [
        pl.col(src)
        .cum_sum()
        .shift(1)
        .fill_null(0)
        .over(["coach", "season"], order_by=["kickoff", "game_id"])
        .alias(f"season_{name}")
        for name, src in metrics.items()
    ]
    out = ordered.with_columns(career + season)
    return out.with_columns(
        (pl.col("career_wins") + pl.col("career_losses") + pl.col("career_ties")).alias("career_games"),
        (pl.col("season_wins") + pl.col("season_losses") + pl.col("season_ties")).alias("season_games"),
    )


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def load_schedules_frame(seasons: Iterable[int] | None = None, root: Path | None = None) -> pl.DataFrame:
    """Read the ingested ``nflverse/schedules`` partitions off disk.

    Reads **all** available seasons regardless of ``seasons``: career records
    need the full history. ``seasons`` only narrows what gets written.
    """
    from nfl.config import RAW_DIR
    from nfl.data.nflverse import raw_table_dir

    del seasons
    directory = raw_table_dir("schedules", root or RAW_DIR)
    if not any(directory.rglob("*.parquet")):
        raise FileNotFoundError(
            f"no schedules partitions under {directory}; run `nfl ingest nflverse "
            f"--tables schedules` first"
        )
    return pl.read_parquet(str(directory / "**" / "*.parquet"), hive_partitioning=True)


def ingest_coach_records(
    seasons: Iterable[int],
    *,
    force: bool = False,
    raw_root: Path | None = None,
    processed_root: Path | None = None,
    written: list[int] | None = None,
) -> Path:
    """Build ``coach_records`` for ``seasons`` and write season partitions.

    The build always reads every ingested schedule season — career counts are
    cumulative — and then writes only the requested seasons. Idempotent: an
    existing partition is left alone unless ``force``. Pass ``written`` to learn
    which seasons were actually rewritten.
    """
    season_list = sorted(set(seasons))
    processed_root = PROCESSED_DIR if processed_root is None else Path(processed_root)
    targets = [
        s
        for s in season_list
        if force or not coach_records_path(s, processed_root).exists()
    ]
    if targets:
        schedules = load_schedules_frame(root=raw_root)
        records = build_coach_records(schedules)
        for season in targets:
            part = records.filter(pl.col("season") == season)
            if part.height == 0:
                continue
            path = coach_records_path(season, processed_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            part.write_parquet(path, compression="zstd")
            if written is not None:
                written.append(season)
    return processed_root / TABLE
