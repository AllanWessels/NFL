"""A small, real assembled dataset for the model tests.

Not a test module — pytest never collects it. It builds the frame the model
tests fit on, by running the *real* :func:`~nfl.features.assemble.assemble_dataset`
over the committed BAL @ KC play-by-play fixture. That gives the genuine 775
feature columns, the genuine dtype mix (float, int, bool, string) and the
genuine null pattern, which is the whole point: a model that only ever sees a
hand-made numeric frame in its tests will meet all three for the first time in
production.

One real game has one outcome, and a logistic regression cannot be fitted to a
single class, so :func:`assembled_games` pairs it with a **mirror**: the same
plays under a new game id with the home/away margin negated and the label
flipped. Two games, both outcomes, every column real.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import polars as pl

from nfl.features.assemble import assemble_dataset, write_dataset

FIXTURES = Path(__file__).parent / "fixtures"
PBP_PATH = FIXTURES / "nflverse" / "pbp_2024_01_BAL_KC.parquet"
SCHEDULES_PATH = FIXTURES / "nflverse" / "schedules_2024.parquet"

#: Columns whose sign is flipped when a game is mirrored, so the mirror's
#: features still point at its (flipped) label rather than contradicting it.
_MIRRORED_COLUMNS = (
    "bi__score_diff_home",
    "bi__score_diff_posteam",
    "bi__timeout_diff_home",
    "ig__score_diff_home",
)


@cache
def assembled_games(season: int = 2024) -> pl.DataFrame:
    """Two assembled games — one real, one mirrored — stamped with ``season``.

    Cached: assembling runs every feature builder, and several tests want the
    same frame. Treat the result as read-only.
    """
    pbp = pl.read_parquet(PBP_PATH)
    schedules = pl.read_parquet(SCHEDULES_PATH)
    real = assemble_dataset([2024], pbp=pbp, schedules=schedules)
    flips = [
        (-pl.col(c).cast(pl.Float64)).cast(real.schema[c]).alias(c)
        for c in _MIRRORED_COLUMNS
        if c in real.columns
    ]
    mirror = real.with_columns(
        pl.concat_str([pl.lit("mirror_"), pl.col("meta__game_id")]).alias("meta__game_id"),
        (1.0 - pl.col("label__home_win")).alias("label__home_win"),
        *flips,
    )
    both = pl.concat([real, mirror])
    return both.with_columns(
        pl.lit(season, dtype=both.schema["meta__season"]).alias("meta__season"),
        pl.concat_str([pl.lit(f"{season}_"), pl.col("meta__game_id")]).alias("meta__game_id"),
    )


def assembled_seasons(seasons: tuple[int, ...] = (2021, 2022, 2023, 2024)) -> pl.DataFrame:
    """The two games replicated across ``seasons`` — enough to exercise the splits."""
    return pl.concat([assembled_games(s) for s in seasons], how="vertical_relaxed")


def write_tiny_dataset(
    processed_root: Path, seasons: tuple[int, ...] = (2021, 2022, 2023, 2024)
) -> Path:
    """Write :func:`assembled_seasons` to ``<processed_root>/dataset`` and return it."""
    write_dataset(assembled_seasons(seasons), root=processed_root)
    return processed_root / "dataset"
