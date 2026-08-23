"""Offline backtesting — run a model over a dataset frame, emit ledger rows.

The output is a frame in :data:`nfl.logging.ledger.LEDGER_SCHEMA`, already
resolved (the dataset carries ``label__home_win``), so
:mod:`nfl.logging.evaluate` scores a backtest and a season of live predictions
with the same code and the same definitions. That equivalence is the reason this
module exists at all: without it, "offline log loss" and "production log loss"
are two numbers computed by two functions and any gap between them is
unattributable.

    >>> from nfl.features.assemble import read_dataset
    >>> from nfl.logging.backtest import backtest
    >>> frame = backtest(model, read_dataset(2024), slices=["q4"])   # doctest: +SKIP
    >>> report = build_report(frame, source="backtest")              # doctest: +SKIP

Market columns are left null. A backtest frame gains market prices the same way
a ledger frame does — :func:`nfl.logging.evaluate.attach_closing_lines`, after
the predictions exist (ADR 0001).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from polars.datatypes import DataType, DataTypeClass

from nfl.data.market_blocklist import assert_no_market_columns
from nfl.features.assemble import LABEL_COLUMN, feature_columns
from nfl.logging import ledger as ledger_mod

log = logging.getLogger(__name__)

__all__ = [
    "SLICES",
    "backtest",
    "resolve_slices",
]

SliceFn = Callable[[pl.DataFrame], pl.DataFrame]


def _q4_at(seconds: int) -> SliceFn:
    from nfl.features.splits import q4_at

    def fn(df: pl.DataFrame) -> pl.DataFrame:
        return q4_at(df, seconds)

    return fn


def _q4_window(df: pl.DataFrame) -> pl.DataFrame:
    from nfl.features.splits import q4_window

    return q4_window(df, 0, 900)


def _last_two_minutes(df: pl.DataFrame) -> pl.DataFrame:
    from nfl.features.splits import q4_window

    return q4_window(df, 0, 120)


#: Named row filters. ``backtest(slices=...)`` takes the *union* of the named
#: slices, so a sweep can score "Q4 plus the three marks" in one pass instead of
#: re-running the model per slice.
SLICES: dict[str, SliceFn] = {
    "all": lambda df: df,
    "q4": _q4_window,
    "q4_last2": _last_two_minutes,
    "q4_600": _q4_at(600),
    "q4_300": _q4_at(300),
    "q4_120": _q4_at(120),
}


def resolve_slices(
    slices: Sequence[str] | Mapping[str, SliceFn] | None,
) -> dict[str, SliceFn]:
    """Normalise the ``slices`` argument to ``{name: filter}``."""
    if slices is None:
        return {"all": SLICES["all"]}
    if isinstance(slices, Mapping):
        return dict(slices)
    unknown = [s for s in slices if s not in SLICES]
    if unknown:
        raise KeyError(f"unknown slice(s) {unknown}; known: {sorted(SLICES)}")
    return {name: SLICES[name] for name in slices}


def _select_rows(
    frame: pl.DataFrame, slices: Mapping[str, SliceFn]
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Union of the named slices, de-duplicated on ``(game_id, play_id)``."""
    counts: dict[str, int] = {}
    parts: list[pl.DataFrame] = []
    for name, fn in slices.items():
        part = fn(frame)
        counts[name] = part.height
        parts.append(part)
    if not parts:
        return frame.head(0), counts
    if len(parts) == 1:
        # One slice: pass its rows straight through. De-duplicating here would
        # silently drop genuinely repeated rows a caller handed us on purpose.
        return parts[0], counts
    out = pl.concat(parts, how="vertical_relaxed")
    keys = [c for c in ("meta__game_id", "meta__play_id") if c in out.columns]
    if keys:
        out = out.unique(subset=keys, keep="first", maintain_order=True)
    return out, counts


def _row_hashes(features: pl.DataFrame, names: Sequence[str]) -> list[str]:
    return [ledger_mod.feature_hash(row, names) for row in features.iter_rows()]


def backtest(
    model: Any,
    dataset_frame: pl.DataFrame,
    slices: Sequence[str] | Mapping[str, SliceFn] | None = None,
    *,
    model_name: str | None = None,
    model_version: str = "backtest",
    model_kind: str = "baseline",
    confidence: float | np.ndarray | None = None,
    confidence_method: str = "ensemble_spread",
    source: str = "backtest",
    feature_set_version: str = "1",
    hash_rows: bool = False,
    batch_size: int | None = None,
    write: bool = False,
    root: Path | None = None,
) -> pl.DataFrame:
    """Run ``model`` over ``dataset_frame`` and return ledger-schema rows.

    ``model`` needs only the :class:`~nfl.models.base.WinProbModel` surface used
    here — ``predict_proba(X) -> np.ndarray`` of ``P(home win)``, plus an
    optional ``name``. Features are selected with
    :func:`nfl.features.assemble.feature_columns`, so no ``meta__``/``label__``
    column ever reaches the model, and
    :func:`~nfl.data.market_blocklist.assert_no_market_columns` runs on the
    frame that does.

    ``hash_rows`` computes a per-row SHA-256 of the ordered feature vector — the
    DATA_MODEL §13 reproducibility anchor, but O(rows x features) in Python. Off
    by default: the frame-level hash of the ordered *feature-set* still lands in
    ``feature_hash``, which is enough to detect a feature-set change, just not a
    single perturbed value.

    ``write=True`` appends the result to the real ledger. Off by default — a
    parameter sweep should not pollute the production ledger with a hundred
    throwaway versions.
    """
    names = feature_columns(dataset_frame)
    if not names:
        raise ValueError("dataset frame has no feature columns; run `nfl features build`")

    selected, counts = _select_rows(dataset_frame, resolve_slices(slices))
    if selected.is_empty():
        log.warning("backtest: no rows selected (slice counts=%s)", counts)
        return ledger_mod.empty_ledger()

    features = selected.select(names)
    assert_no_market_columns(features, context="backtest")
    leaked = [c for c in features.columns if c.startswith(("meta__", "label__"))]
    if leaked:  # pragma: no cover - feature_columns excludes these by construction
        raise AssertionError(f"meta/label columns reached the model: {leaked}")

    started = time.perf_counter()
    probs = _predict(model, features, batch_size)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if probs.shape[0] != selected.height:
        raise ValueError(
            f"model returned {probs.shape[0]} probabilities for {selected.height} rows"
        )
    per_row_ms = elapsed_ms / max(selected.height, 1)

    set_hash = ledger_mod.feature_hash((), names)
    hashes = _row_hashes(features, names) if hash_rows else None

    nonnull = features.select(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in names]).alias("n")
    ).get_column("n")

    name = model_name or getattr(model, "name", None) or type(model).__name__
    now = datetime.now(UTC)

    out = _assemble(
        selected,
        probs=probs,
        confidence=confidence,
        nonnull=nonnull,
        n_features=len(names),
        hashes=hashes,
        set_hash=set_hash,
        model_name=name,
        model_version=model_version,
        model_kind=model_kind,
        confidence_method=confidence_method,
        source=source,
        feature_set_version=feature_set_version,
        per_row_ms=per_row_ms,
        now=now,
    )
    log.info(
        "backtest: %s %s over %d rows in %.1fms (%s)",
        name,
        model_version,
        out.height,
        elapsed_ms,
        counts,
    )
    if write:
        ledger_mod.append(out, root=root)
    return out


def _predict(model: Any, features: pl.DataFrame, batch_size: int | None) -> np.ndarray:
    """``predict_proba`` in one shot, or in batches when asked.

    Batching exists for the TF nets on a 16 GB card: 770k x 775 float32 is ~2.4
    GB before the model allocates anything.
    """
    if batch_size is None or batch_size >= features.height:
        return np.asarray(model.predict_proba(features), dtype=float).ravel()
    chunks = [
        np.asarray(model.predict_proba(features.slice(i, batch_size)), dtype=float).ravel()
        for i in range(0, features.height, batch_size)
    ]
    return np.concatenate(chunks)


def _meta(frame: pl.DataFrame, name: str, dtype: DataType | DataTypeClass) -> pl.Expr:
    """``meta__<name>`` or ``<name>`` off the dataset, else a typed null."""
    for candidate in (f"meta__{name}", name):
        if candidate in frame.columns:
            return pl.col(candidate).cast(dtype, strict=False)
    return pl.lit(None, dtype=dtype)


def _assemble(
    selected: pl.DataFrame,
    *,
    probs: np.ndarray,
    confidence: float | np.ndarray | None,
    nonnull: pl.Series,
    n_features: int,
    hashes: list[str] | None,
    set_hash: str,
    model_name: str,
    model_version: str,
    model_kind: str,
    confidence_method: str,
    source: str,
    feature_set_version: str,
    per_row_ms: float,
    now: datetime,
) -> pl.DataFrame:
    """Ledger rows from the selected dataset rows plus the model output."""
    n = selected.height
    if confidence is None:
        # No ensemble to disagree with: confidence is distance from the coin
        # flip, which is honest for a single model and never claims more.
        conf = np.abs(probs - 0.5) * 2.0
    elif np.isscalar(confidence):
        conf = np.full(n, float(confidence))  # type: ignore[arg-type]
    else:
        conf = np.asarray(confidence, dtype=float).ravel()

    ids = [ledger_mod.new_prediction_id() for _ in range(n)]
    label = (
        pl.col(LABEL_COLUMN).cast(pl.Boolean, strict=False)
        if LABEL_COLUMN in selected.columns
        else pl.lit(None, dtype=pl.Boolean)
    )
    clock = _meta(selected, "game_seconds_remaining", pl.Int32)

    frame = selected.with_columns(
        pl.Series("prediction_id", ids),
        pl.Series("p_home", probs),
        pl.Series("confidence", np.clip(conf, 0.0, 1.0)),
        pl.Series("features_nonnull_count", nonnull.cast(pl.Int32)),
    ).select(
        pl.col("prediction_id"),
        pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("predicted_at_utc"),
        _meta(selected, "game_id", pl.String).alias("game_id"),
        _meta(selected, "season", pl.Int32).alias("season"),
        _meta(selected, "week", pl.Int32).alias("week"),
        _meta(selected, "home_team", pl.String).alias("home_team"),
        _meta(selected, "away_team", pl.String).alias("away_team"),
        pl.lit(source).alias("source"),
        pl.lit(False).alias("is_live"),
        _meta(selected, "play_id", pl.Int64).alias("snapshot_play_index"),
        _meta(selected, "period", pl.Int32).alias("snapshot_period"),
        clock.alias("snapshot_game_seconds_remaining"),
        _meta(selected, "home_score", pl.Int32).alias("home_score"),
        _meta(selected, "away_score", pl.Int32).alias("away_score"),
        _meta(selected, "posteam", pl.String).alias("posteam"),
        _meta(selected, "down", pl.Int32).alias("down"),
        _meta(selected, "ydstogo", pl.Int32).alias("ydstogo"),
        _meta(selected, "yardline_100", pl.Int32).alias("yardline_100"),
        pl.lit(model_name).alias("model_name"),
        pl.lit(model_version).alias("model_version"),
        pl.lit(model_kind).alias("model_kind"),
        pl.col("p_home"),
        (1.0 - pl.col("p_home")).alias("p_away"),
        pl.col("confidence"),
        pl.lit(confidence_method).alias("confidence_method"),
        (
            pl.Series("feature_hash", hashes)
            if hashes is not None
            else pl.lit(set_hash).alias("feature_hash")
        ),
        pl.lit(feature_set_version).alias("feature_set_version"),
        pl.col("features_nonnull_count"),
        pl.lit(n_features, dtype=pl.Int32).alias("features_total_count"),
        pl.lit(ledger_mod.git_sha()).alias("git_sha"),
        pl.lit(per_row_ms).alias("latency_ms"),
        pl.lit(ledger_mod.SCHEMA_VERSION).alias("schema_version"),
        pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("ingested_at"),
        label.alias("resolved_home_win"),
        pl.lit(now).cast(pl.Datetime("us", "UTC")).alias("resolved_at_utc"),
    )

    # Backtest labels come from the dataset, so brier/log_loss can be filled now.
    eps = 1e-15
    p = pl.col("p_home").clip(eps, 1.0 - eps)
    y = pl.col("resolved_home_win").cast(pl.Float64)
    frame = frame.with_columns(
        pl.when(y.is_null()).then(None).otherwise((p - y) ** 2).alias("brier"),
        pl.when(y.is_null())
        .then(None)
        .otherwise(-(y * p.log() + (1.0 - y) * (1.0 - p).log()))
        .alias("log_loss"),
        pl.when(pl.col("resolved_home_win").is_null())
        .then(None)
        .otherwise(pl.col("resolved_at_utc"))
        .alias("resolved_at_utc"),
    )
    return ledger_mod.conform(frame)
