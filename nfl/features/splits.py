"""Season splits and the late-game slices the project is actually judged on.

Two ideas live here.

**Season-based splits.** Time-ordered, never random: train on seasons up to
``train_through`` (default 2022), validate on ``val_season`` (2023), test on
``test_from`` and later (2024+). A random split across plays would leak — plays
from the same game would land on both sides — and a random split across games
would still leak the era. Seasons are the only honest cut.

**Q4 slices.** The target use case is a snapshot late in the fourth quarter, so
model quality is reported on those rows specifically. :func:`q4_at` takes the
one row per game that represents the state at (or immediately before) a given
mark; :func:`q4_window` takes every row inside a window.

:func:`not_collapsed` is the placeholder for the downstream trading filter —
"only look at games the market has *not* already resolved past ~85/15". The
probability column it keys on is market data, so it never exists on a feature
frame; it is joined in at evaluation time (ADR 0001), and until Phase 6 lands
that join this function passes frames through untouched.

Column names are resolved leniently: ``meta__game_seconds_remaining`` on an
assembled dataset, plain ``game_seconds_remaining`` on a raw play-by-play
frame — both work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import polars as pl

__all__ = [
    "COLLAPSE_BOUNDS",
    "COLLAPSE_PROB_COLUMN",
    "DEFAULT_SPLIT",
    "REGULATION_Q4_SECONDS",
    "SeasonSplit",
    "Splits",
    "add_split_column",
    "holdout_frame",
    "not_collapsed",
    "q4_at",
    "q4_window",
    "season_of",
    "split_frames",
    "test_frame",
    "train_frame",
    "val_frame",
]

#: Seconds remaining in regulation at the start of the fourth quarter.
REGULATION_Q4_SECONDS = 900

#: The market probability column :func:`not_collapsed` keys on. Market data —
#: evaluation frames only, never a feature (ADR 0001).
COLLAPSE_PROB_COLUMN = "market_prob_home"

#: Outside these bounds the market is treated as collapsed (no tradeable edge).
COLLAPSE_BOUNDS: tuple[float, float] = (0.15, 0.85)

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SeasonSplit:
    """Which seasons go where. Defaults match CLAUDE.md / the CLI defaults."""

    train_through: int = 2022
    val_season: int = 2023
    test_from: int = 2024

    def __post_init__(self) -> None:
        if self.val_season <= self.train_through:
            raise ValueError(
                f"val_season {self.val_season} must be after train_through {self.train_through}"
            )
        if self.test_from <= self.val_season:
            raise ValueError(
                f"test_from {self.test_from} must be after val_season {self.val_season}"
            )

    def name_for(self, season: int | None) -> SplitName | None:
        """Which split a season belongs to, or ``None`` if it belongs to none."""
        if season is None:
            return None
        if season <= self.train_through:
            return "train"
        if season == self.val_season:
            return "val"
        if season >= self.test_from:
            return "test"
        return None


#: train <= 2022, val = 2023, test >= 2024.
DEFAULT_SPLIT = SeasonSplit()


class Splits(NamedTuple):
    """The three frames, in time order."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def _resolve(df: pl.DataFrame, name: str, *, required: bool = True) -> str | None:
    """Find ``meta__<name>`` or ``<name>`` on ``df``."""
    for candidate in (f"meta__{name}", name):
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(f"frame has neither 'meta__{name}' nor {name!r}; columns={df.columns}")
    return None


def season_of(df: pl.DataFrame) -> pl.Expr:
    """The season column as an expression, whatever it is called."""
    return pl.col(str(_resolve(df, "season")))


def _season_filter(df: pl.DataFrame, predicate: pl.Expr) -> pl.DataFrame:
    return df.filter(predicate)


def train_frame(df: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> pl.DataFrame:
    """Rows from seasons at or before ``split.train_through``."""
    return _season_filter(df, season_of(df) <= split.train_through)


def val_frame(df: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> pl.DataFrame:
    """Rows from ``split.val_season``."""
    return _season_filter(df, season_of(df) == split.val_season)


def test_frame(df: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> pl.DataFrame:
    """Rows from ``split.test_from`` onwards."""
    return _season_filter(df, season_of(df) >= split.test_from)


#: Alias for :func:`test_frame`. Importing ``test_frame`` into a test module
#: makes pytest try to *collect* it; import ``holdout_frame`` there instead.
holdout_frame = test_frame


def split_frames(df: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> Splits:
    """``(train, val, test)`` for ``df``. Seasons in no split are dropped."""
    return Splits(train_frame(df, split), val_frame(df, split), test_frame(df, split))


def add_split_column(
    df: pl.DataFrame, split: SeasonSplit = DEFAULT_SPLIT, *, name: str = "meta__split"
) -> pl.DataFrame:
    """Tag every row ``train``/``val``/``test`` (null when it belongs to none).

    ``meta__``-prefixed by default so the tag can never be read as a feature.
    """
    season = season_of(df)
    return df.with_columns(
        pl.when(season <= split.train_through)
        .then(pl.lit("train"))
        .when(season == split.val_season)
        .then(pl.lit("val"))
        .when(season >= split.test_from)
        .then(pl.lit("test"))
        .otherwise(None)
        .cast(pl.String)
        .alias(name)
    )


# --------------------------------------------------------------------------- #
# Q4 slices
# --------------------------------------------------------------------------- #


def _q4_candidates(df: pl.DataFrame, *, include_overtime: bool) -> tuple[pl.DataFrame, str]:
    gsr = str(_resolve(df, "game_seconds_remaining"))
    out = df.filter(pl.col(gsr).is_not_null() & (pl.col(gsr) <= REGULATION_Q4_SECONDS))
    period = _resolve(df, "period", required=False) or _resolve(df, "qtr", required=False)
    if period is not None:
        out = out.filter(pl.col(period) >= 4 if include_overtime else pl.col(period) == 4)
    return out, gsr


def q4_at(
    df: pl.DataFrame,
    seconds_remaining: int,
    *,
    tolerance: int | None = None,
    include_overtime: bool = False,
) -> pl.DataFrame:
    """One row per game: the state at ``seconds_remaining`` left in regulation.

    "At" means *the last snapshot no later than that mark* — the play whose
    ``game_seconds_remaining`` is the smallest value still ``>=`` the target, so
    the row genuinely predates the moment asked about. Games that never reach
    the mark (or lack the clock) contribute nothing. ``tolerance`` caps how far
    before the mark the chosen row may be, in seconds.
    """
    if not 0 <= seconds_remaining <= REGULATION_Q4_SECONDS:
        raise ValueError(
            f"seconds_remaining must be within the fourth quarter "
            f"(0..{REGULATION_Q4_SECONDS}); got {seconds_remaining}"
        )
    cand, gsr = _q4_candidates(df, include_overtime=include_overtime)
    cand = cand.filter(pl.col(gsr) >= seconds_remaining)
    if tolerance is not None:
        cand = cand.filter((pl.col(gsr) - seconds_remaining) <= tolerance)
    game = str(_resolve(cand, "game_id"))
    play = _resolve(cand, "play_id", required=False)
    by = [game, gsr] + ([play] if play else [])
    descending = [False, False] + ([True] if play else [])
    return cand.sort(by, descending=descending).unique(
        subset=[game], keep="first", maintain_order=True
    )


def q4_window(
    df: pl.DataFrame, lo: int, hi: int, *, include_overtime: bool = False
) -> pl.DataFrame:
    """Every row whose clock sits in ``[lo, hi]`` seconds remaining, Q4 only.

    Bounds are seconds *remaining*, so ``q4_window(df, 0, 300)`` is the last
    five minutes. The order of the two bounds does not matter.
    """
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    cand, gsr = _q4_candidates(df, include_overtime=include_overtime)
    return cand.filter(pl.col(gsr).is_between(low, high))


# --------------------------------------------------------------------------- #
# Market-collapse filter (evaluation only)
# --------------------------------------------------------------------------- #


def not_collapsed(
    df: pl.DataFrame,
    *,
    prob_col: str = COLLAPSE_PROB_COLUMN,
    bounds: Sequence[float] = COLLAPSE_BOUNDS,
    missing: Literal["keep", "raise"] = "keep",
) -> pl.DataFrame:
    """Keep rows where the market has not already resolved the game.

    **Placeholder.** The column it keys on is a market implied probability,
    which by ADR 0001 never exists on a feature frame — it is joined in at
    evaluation time from ``nfl.data.market``. Until that join lands (Phase 6),
    calling this on a feature frame returns the frame unchanged; pass
    ``missing="raise"`` in evaluation code that must not silently no-op.

    Rows with a null probability are kept: unknown is not the same as
    collapsed.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"bounds must satisfy 0 <= lo < hi <= 1; got {(lo, hi)}")
    if prob_col not in df.columns:
        if missing == "raise":
            raise KeyError(
                f"{prob_col!r} not on the frame; join the market snapshot before filtering"
            )
        return df
    prob = pl.col(prob_col).cast(pl.Float64, strict=False)
    return df.filter(prob.is_null() | prob.is_between(lo, hi))
