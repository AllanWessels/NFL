"""Leakage checks shared by every feature builder and by the assembled dataset.

Three of the four failure modes in ``docs/DATA_MODEL.md`` ("Leakage taxonomy")
are mechanically checkable, and they are checked here so that each builder gets
the same check rather than its own approximation of it:

:func:`check_delete_the_future`
    Rebuild the feature frame with later data deleted. Rows that survive the
    truncation must be **bit-identical** to what the full build produced. This
    catches post-play state at play *t* (taxonomy 1) and end-of-season ranks
    used mid-season (taxonomy 2) with one test, because both show up as a row
    that changes when the future is removed.
:func:`check_no_market`
    The market blocklist over a finished frame (taxonomy 4).
:func:`check_no_trivial_leak`
    Dataset-level: no feature may be a near-perfect stand-in for the label. A
    feature correlating |r| > 0.999 with ``label__home_win`` is a copy of the
    answer, not a signal — usually a postgame column that slipped in.

The functions raise :class:`LeakageError` on failure and return a small result
object on success, so a test reads as ``check_...(...)`` with no assertion of
its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from nfl.data.market_blocklist import MarketColumnLeak, assert_no_market_columns, market_columns_in

__all__ = [
    "DEFAULT_LABEL",
    "LeakSuspect",
    "LeakageError",
    "TruncationCheck",
    "check_delete_the_future",
    "check_no_market",
    "check_no_trivial_leak",
    "truncate_inputs",
]

DEFAULT_LABEL = "label__home_win"

#: Key columns to align on, most specific first.
_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("game_id", "play_id"),
    ("game_id", "team"),
    ("meta__game_id", "meta__play_id"),
    ("game_id",),
    ("meta__game_id",),
)


class LeakageError(AssertionError):
    """A feature frame used information it could not have had at prediction time."""


@dataclass(frozen=True, slots=True)
class TruncationCheck:
    """What :func:`check_delete_the_future` actually compared."""

    rows_compared: int
    columns_compared: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.rows_compared > 0


def truncate_inputs(
    inputs: Mapping[str, pl.DataFrame],
    cut: Any,
    *,
    order_col: str = "play_id",
    inclusive: bool = True,
    only: Sequence[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Delete the future from every input frame that has ``order_col``.

    ``inclusive`` keeps rows *at* the cut (the natural reading for a play id);
    pass ``inclusive=False`` for a kickoff timestamp, where a pre-game feature
    may use only strictly earlier games. Frames without ``order_col`` — and
    frames not named in ``only`` — pass through untouched.
    """
    out: dict[str, pl.DataFrame] = {}
    for name, frame in inputs.items():
        if only is not None and name not in only:
            out[name] = frame
            continue
        if order_col not in frame.columns:
            out[name] = frame
            continue
        col = pl.col(order_col)
        out[name] = frame.filter(col <= cut if inclusive else col < cut)
    return out


def _pick_keys(df: pl.DataFrame, keys: Sequence[str] | None) -> tuple[str, ...]:
    if keys is not None:
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise KeyError(f"key column(s) {missing} not in builder output {df.columns}")
        return tuple(keys)
    for candidate in _KEY_CANDIDATES:
        if all(c in df.columns for c in candidate):
            return candidate
    raise KeyError(f"cannot infer key columns for a frame with columns {df.columns}")


def _differing_columns(
    full: pl.DataFrame, truncated: pl.DataFrame, keys: Sequence[str], atol: float
) -> dict[str, int]:
    """``{column: rows that disagree}`` between the two builds, aligned on ``keys``."""
    cols = [c for c in truncated.columns if c in full.columns and c not in keys]
    joined = full.join(truncated.select([*keys, *cols]), on=list(keys), how="inner", suffix="__t")
    if joined.height == 0:
        return {}
    diffs: dict[str, int] = {}
    for col in cols:
        left, right = pl.col(col), pl.col(f"{col}__t")
        if joined.schema[col].is_numeric() and joined.schema[f"{col}__t"].is_numeric():
            expr = (left.is_null() != right.is_null()) | (
                (left.cast(pl.Float64) - right.cast(pl.Float64)).abs() > atol
            ).fill_null(False)
        else:
            expr = left.ne_missing(right)
        bad = int(joined.select(expr.fill_null(True).sum()).item())
        if bad:
            diffs[col] = bad
    return diffs


def check_delete_the_future(
    builder_fn: Callable[..., pl.DataFrame] | Any,
    inputs: Mapping[str, pl.DataFrame],
    cut: Any,
    *,
    order_col: str = "play_id",
    inclusive: bool = True,
    keys: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
    atol: float = 1e-9,
) -> TruncationCheck:
    """Assert that deleting data after ``cut`` does not change the rows before it.

    ``builder_fn`` is a ``build_*`` function or anything with a ``build``
    method (a :class:`~nfl.features.registry.BuilderSpec`, for instance).
    ``inputs`` are the frames it takes, by keyword name. Every input carrying
    ``order_col`` is truncated at ``cut``, the builder is re-run, and the
    surviving rows are compared column by column against the full build.

    Raises :class:`LeakageError` naming the offending columns; returns what was
    compared so a test can assert the check was not vacuous.
    """
    fn = builder_fn.build if hasattr(builder_fn, "build") else builder_fn
    full = fn(**dict(inputs))
    truncated = fn(
        **truncate_inputs(inputs, cut, order_col=order_col, inclusive=inclusive, only=only)
    )

    key_cols = _pick_keys(full, keys)
    missing_keys = [k for k in key_cols if k not in truncated.columns]
    if missing_keys:
        raise LeakageError(f"truncated build lost key column(s) {missing_keys}")

    extra = truncated.join(full.select(list(key_cols)), on=list(key_cols), how="anti")
    if extra.height:
        raise LeakageError(
            f"truncated build produced {extra.height} row(s) the full build did not; "
            f"the builder is not a pure function of its inputs"
        )

    diffs = _differing_columns(full, truncated, key_cols, atol)
    aligned = full.join(truncated.select(list(key_cols)), on=list(key_cols), how="semi").height
    if diffs:
        detail = ", ".join(f"{c} ({n} row(s))" for c, n in sorted(diffs.items()))
        raise LeakageError(
            f"deleting data after {order_col} <= {cut!r} changed earlier rows: {detail}. "
            f"A feature for play t must use only plays < t (docs/DATA_MODEL.md)."
        )
    compared = tuple(c for c in truncated.columns if c in full.columns and c not in key_cols)
    return TruncationCheck(rows_compared=aligned, columns_compared=compared, keys=key_cols)


def check_no_market(df: Any, *, context: str | None = None) -> list[str]:
    """Raise :class:`~nfl.data.market_blocklist.MarketColumnLeak` if ``df`` has market columns.

    Returns the (empty) offender list on success so a caller can log it.
    """
    assert_no_market_columns(df, context=context)
    return market_columns_in(df)


@dataclass(frozen=True, slots=True)
class LeakSuspect:
    """A feature that tracks the label too closely to be real."""

    column: str
    correlation: float
    scope: str  # "pooled" | "per_game"
    rows: int = 0

    def __str__(self) -> str:
        return f"{self.column} (r={self.correlation:+.6f}, {self.scope}, n={self.rows})"


@dataclass(slots=True)
class LeakReport:
    """Result of :func:`check_no_trivial_leak`."""

    checked: tuple[str, ...] = ()
    suspects: tuple[LeakSuspect, ...] = ()
    skipped: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return not self.suspects


def _corr(frame: pl.DataFrame, col: str, label_col: str) -> tuple[float, int] | None:
    sub = frame.select(
        pl.col(col).cast(pl.Float64, strict=False).alias("x"),
        pl.col(label_col).cast(pl.Float64, strict=False).alias("y"),
    ).drop_nulls()
    if sub.height < 3:
        return None
    if sub["x"].n_unique() < 2 or sub["y"].n_unique() < 2:
        return None
    value = sub.select(pl.corr("x", "y")).item()
    if value is None:
        return None
    return float(value), sub.height


def check_no_trivial_leak(
    df: pl.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL,
    threshold: float = 0.999,
    features: Sequence[str] | None = None,
    game_col: str | None = None,
    raise_on_leak: bool = True,
) -> LeakReport:
    """Assert no feature is a near-copy of the label.

    Two scopes, because the label is constant within a game and each catches a
    different mistake:

    * **pooled** — over every play. Catches a postgame column broadcast onto
      every row (final score, ``result``, a third-party win probability at the
      final whistle).
    * **per_game** — over the *last* row of each game. Catches a feature that is
      only a giveaway at the end of the game, which pooled correlation dilutes.

    A correlation of |r| > ``threshold`` in either scope raises
    :class:`LeakageError`. Non-numeric and constant columns are skipped with a
    recorded reason.
    """
    if label_col not in df.columns:
        raise KeyError(f"{label_col!r} not on the frame; columns={df.columns}")
    from nfl.features.assemble import feature_columns

    cols = list(features) if features is not None else feature_columns(df)
    report = LeakReport()
    checked: list[str] = []
    suspects: list[LeakSuspect] = []
    skipped: dict[str, str] = {}

    game = game_col or next((c for c in ("meta__game_id", "game_id") if c in df.columns), None)
    per_game: pl.DataFrame | None = None
    if game is not None:
        play = next((c for c in ("meta__play_id", "play_id") if c in df.columns), None)
        ordered = df.sort([game, play]) if play else df
        per_game = ordered.group_by(game).last()

    for col in cols:
        if not df.schema[col].is_numeric():
            skipped[col] = "non-numeric"
            continue
        checked.append(col)
        for scope, frame in (("pooled", df), ("per_game", per_game)):
            if frame is None:
                continue
            result = _corr(frame, col, label_col)
            if result is None:
                skipped.setdefault(col, f"constant or too few rows ({scope})")
                continue
            value, rows = result
            if abs(value) > threshold:
                suspects.append(LeakSuspect(col, value, scope, rows))

    report.checked = tuple(checked)
    report.suspects = tuple(suspects)
    report.skipped = skipped
    if suspects and raise_on_leak:
        detail = "; ".join(str(s) for s in suspects)
        raise LeakageError(
            f"feature(s) correlate with {label_col} above |r| > {threshold}: {detail}. "
            f"A feature that predicts the label perfectly is the label."
        )
    return report


# Re-exported so a builder's test can catch both failure modes from one import.
MarketLeak = MarketColumnLeak
