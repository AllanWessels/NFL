"""Season splits and the late-game slices the project is actually judged on.

Two ideas live here.

**Season-based splits.** Time-ordered, never random: train on seasons up to
``train_through``, validate on ``val_season``, test on ``test_from`` and later.
A random split across plays would leak — plays from the same game would land on
both sides — and a random split across games would still leak the era. Seasons
are the only honest cut.

The boundaries are **rolling**, not pinned to a year someone typed once. ``L``
is the latest season whose *regular season* has finished on disk
(:func:`latest_complete_season`, read from the nflverse schedules), and the
split is :meth:`SeasonSplit.rolling`: **train <= L-2, val L-1, test L**. A
pinned 2022/2023/2024 quietly turns into "train on everything, validate on
nothing" the moment a season lands, and a model validated on nothing cannot be
gated. Call :func:`default_split` to resolve it, or pass an explicit
:class:`SeasonSplit` (``--split`` on the CLI) to override.

Resolution order for :func:`default_split`: the ``NFL_SEASON_SPLIT`` environment
variable, then the schedules on disk, then :data:`FALLBACK_SPLIT` when there are
no schedules to date the rolling window from.

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

import functools
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import polars as pl

log = logging.getLogger(__name__)

__all__ = [
    "COLLAPSE_BOUNDS",
    "COLLAPSE_PROB_COLUMN",
    "FALLBACK_SPLIT",
    "REGULATION_Q4_SECONDS",
    "SPLIT_ENV_VAR",
    "SeasonSplit",
    "Splits",
    "add_split_column",
    "clear_split_cache",
    "default_split",
    "holdout_frame",
    "latest_complete_season",
    "not_collapsed",
    "q4_at",
    "q4_window",
    "resolve_split",
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
    """Which seasons go where.

    Build it with :meth:`rolling` wherever the answer should follow the data:
    the literal boundaries below exist only so :data:`FALLBACK_SPLIT` has
    something to be.
    """

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

    @classmethod
    def rolling(cls, latest_complete_season: int) -> SeasonSplit:
        """train <= ``L-2``, val ``L-1``, test ``L`` — the project default.

        ``L`` is the latest season whose regular season has finished. The most
        recent complete season is the *test* set precisely because it is the one
        nothing has been tuned against, and the season before it is val, so the
        promotion gate always judges a model on the newest football it is
        allowed to have seen.
        """
        latest = int(latest_complete_season)
        return cls(train_through=latest - 2, val_season=latest - 1, test_from=latest)

    @classmethod
    def parse(cls, spec: str) -> SeasonSplit:
        """``"2023/2024/2025"``, ``"rolling:2025"`` or a bare ``"2025"`` (= rolling).

        The three-number form is ``train_through/val_season/test_from``; commas
        work as well as slashes.
        """
        text = str(spec).strip().lower()
        if not text:
            raise ValueError("empty split spec")
        if text.startswith("rolling"):
            _, _, tail = text.partition(":")
            if not tail.strip():
                raise ValueError("'rolling' needs a season, e.g. 'rolling:2025'")
            return cls.rolling(int(tail))
        parts = [p for p in text.replace(",", "/").split("/") if p.strip()]
        if len(parts) == 1:
            return cls.rolling(int(parts[0]))
        if len(parts) != 3:
            raise ValueError(
                f"split spec {spec!r} must be 'train_through/val_season/test_from', "
                "'rolling:<season>' or a single season"
            )
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    @property
    def label(self) -> str:
        """``"<=2023/2024/2025+"`` — one token for a log line or a table cell."""
        return f"<={self.train_through}/{self.val_season}/{self.test_from}+"

    def to_dict(self) -> dict[str, int]:
        """The split as JSON, for a manifest."""
        return {
            "train_through": self.train_through,
            "val_season": self.val_season,
            "test_from": self.test_from,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SeasonSplit | None:
        """Read :meth:`to_dict` back; ``None`` when the payload is not one."""
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                int(payload["train_through"]),
                int(payload["val_season"]),
                int(payload["test_from"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

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


#: What :func:`default_split` falls back to when there are no schedules on disk
#: to date the rolling window from. It is *not* the project default — it is the
#: historical 2022/2023/2024 cut, kept so a checkout with no ``data/`` still has
#: something coherent to split on.
FALLBACK_SPLIT = SeasonSplit(train_through=2022, val_season=2023, test_from=2024)

#: Set this to a :meth:`SeasonSplit.parse` spec to pin the split process-wide.
#: The test suite uses it so a fixture dataset is not re-cut by whatever seasons
#: happen to be sitting in ``data/raw``.
SPLIT_ENV_VAR = "NFL_SEASON_SPLIT"

#: Where :func:`latest_complete_season` looks, relative to ``data/raw``.
SCHEDULES_GLOB = "nflverse/schedules/**/*.parquet"

#: Regular-season game type in the nflverse schedule. Playoff games are ignored
#: when dating a season: the regular season is what the model is trained on, and
#: waiting for the Super Bowl would stall the split for a month.
REGULAR_SEASON_TYPE = "REG"


def _schedule_frame(root: Path | None = None) -> pl.DataFrame | None:
    """Every nflverse schedule partition on disk, or ``None`` if there are none."""
    from nfl.config import RAW_DIR

    base = Path(root) if root is not None else RAW_DIR
    files = sorted(base.glob(SCHEDULES_GLOB))
    if not files:
        return None
    try:
        return pl.read_parquet([str(f) for f in files])
    except (OSError, pl.exceptions.PolarsError) as exc:  # pragma: no cover - corrupt lake
        log.warning("cannot read schedules under %s: %s", base, exc)
        return None


def _latest_complete(frame: pl.DataFrame) -> int | None:
    """The newest season whose regular-season games all carry a final score."""
    if "season" not in frame.columns:
        return None
    outcome = next((c for c in ("result", "home_score") if c in frame.columns), None)
    if outcome is None:
        return None
    reg = frame
    if "game_type" in frame.columns:
        reg = frame.filter(pl.col("game_type") == REGULAR_SEASON_TYPE)
    if reg.is_empty():
        return None
    by_season = (
        reg.group_by("season")
        .agg(pl.len().alias("n"), pl.col(outcome).is_null().sum().alias("unplayed"))
        .filter((pl.col("n") > 0) & (pl.col("unplayed") == 0))
    )
    if by_season.is_empty():
        return None
    return int(by_season.get_column("season").max())  # type: ignore[arg-type]


@functools.lru_cache(maxsize=8)
def _latest_complete_cached(key: str) -> int | None:
    frame = _schedule_frame(Path(key))
    return None if frame is None else _latest_complete(frame)


def latest_complete_season(
    schedules: pl.DataFrame | None = None, *, root: Path | None = None
) -> int | None:
    """``L``: the newest season whose regular season has finished, or ``None``.

    "Finished" means every ``game_type == "REG"`` row for that season carries a
    result. A season mid-flight has null results for the games not yet played,
    so it is not eligible to be the test set — which is the whole point: the
    rolling split must never validate or test on a partial season.

    Pass ``schedules`` to date a frame already in hand; otherwise the nflverse
    schedule partitions under ``data/raw`` are read (and cached — call
    :func:`clear_split_cache` after an ingest).
    """
    if schedules is not None:
        return _latest_complete(schedules)
    from nfl.config import RAW_DIR

    return _latest_complete_cached(str(Path(root) if root is not None else RAW_DIR))


def clear_split_cache() -> None:
    """Forget the cached schedule scan — call after ingesting a new season."""
    _latest_complete_cached.cache_clear()


def default_split(root: Path | None = None) -> SeasonSplit:
    """The split to use when the caller did not name one.

    ``NFL_SEASON_SPLIT`` wins if it is set (a pin, for tests and for reproducing
    an old run), then :meth:`SeasonSplit.rolling` over
    :func:`latest_complete_season`, then :data:`FALLBACK_SPLIT`.
    """
    pinned = os.environ.get(SPLIT_ENV_VAR, "").strip()
    if pinned:
        try:
            return SeasonSplit.parse(pinned)
        except ValueError as exc:
            raise ValueError(
                f"{SPLIT_ENV_VAR}={pinned!r} is not a valid split spec: {exc}"
            ) from exc
    latest = latest_complete_season(root=root)
    if latest is None:
        log.debug("no schedules on disk; falling back to %s", FALLBACK_SPLIT.label)
        return FALLBACK_SPLIT
    return SeasonSplit.rolling(latest)


def resolve_split(split: SeasonSplit | str | None, root: Path | None = None) -> SeasonSplit:
    """``split`` if given (parsing a spec string), else :func:`default_split`."""
    if split is None:
        return default_split(root)
    if isinstance(split, str):
        return SeasonSplit.parse(split)
    return split


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


def train_frame(df: pl.DataFrame, split: SeasonSplit | None = None) -> pl.DataFrame:
    """Rows from seasons at or before ``split.train_through``."""
    return _season_filter(df, season_of(df) <= resolve_split(split).train_through)


def val_frame(df: pl.DataFrame, split: SeasonSplit | None = None) -> pl.DataFrame:
    """Rows from ``split.val_season``."""
    return _season_filter(df, season_of(df) == resolve_split(split).val_season)


def test_frame(df: pl.DataFrame, split: SeasonSplit | None = None) -> pl.DataFrame:
    """Rows from ``split.test_from`` onwards."""
    return _season_filter(df, season_of(df) >= resolve_split(split).test_from)


#: Alias for :func:`test_frame`. Importing ``test_frame`` into a test module
#: makes pytest try to *collect* it; import ``holdout_frame`` there instead.
holdout_frame = test_frame


def split_frames(df: pl.DataFrame, split: SeasonSplit | None = None) -> Splits:
    """``(train, val, test)`` for ``df``. Seasons in no split are dropped."""
    resolved = resolve_split(split)
    return Splits(train_frame(df, resolved), val_frame(df, resolved), test_frame(df, resolved))


def add_split_column(
    df: pl.DataFrame, split: SeasonSplit | None = None, *, name: str = "meta__split"
) -> pl.DataFrame:
    """Tag every row ``train``/``val``/``test`` (null when it belongs to none).

    ``meta__``-prefixed by default so the tag can never be read as a feature.
    """
    split = resolve_split(split)
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
