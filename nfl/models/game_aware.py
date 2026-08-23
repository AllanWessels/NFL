"""Game-aware training: row weights that count games, and selection on the Q4 slices.

ADR 0003 item 2. Two corrections to "treat every parquet row as an independent
example", both of which the Phase 5 numbers argued for.

**Row weight = 1 / plays-in-game.** A 200-play overtime thriller and a 120-play
blowout carry one outcome each, but the first contributes 200 identical labels
to the loss and the second 120. Unweighted, the fit quietly decides long games
matter 1.7x more, and — worse — the effective sample size stays ~3.4k games
however many rows you feed it. :func:`game_weights` divides each row by its
game's play count, so every game contributes exactly 1.0 and the weight vector
sums to the number of games. XGBoost and Keras both take it as ``sample_weight``
verbatim.

**Selection on Q4, not on overall log-loss.** The project is judged on three
one-row-per-game marks (600s / 300s / 120s left), which together are ~10k rows
out of 130k in a validation season. Early-stopping on overall val log-loss
optimises first-quarter snapshots the product does not care about; measured on
this dataset the two disagree about where the best epoch is.
:func:`q4_selection` picks out those rows once, and

* XGBoost gets them as a fourth ``DMatrix`` (the *last* eval set, which is the
  one ``early_stopping_rounds`` watches) carrying per-mark weights ``1/n_mark``,
  so the booster's own weighted ``logloss`` equals the mean of the three
  per-mark log-losses exactly;
* Keras gets :func:`make_q4_callback`, which runs the same three marks at the
  end of every epoch and writes ``val_q4_log_loss`` into ``logs`` where
  ``EarlyStopping`` and ``ReduceLROnPlateau`` can monitor it.

The metric itself is :meth:`Q4Selection.mean_log_loss` in both cases, so the
XGBoost and Keras runs are selected on the same number.

    >>> import numpy as np, polars as pl
    >>> meta = pl.DataFrame({"meta__game_id": ["a", "a", "b"]})
    >>> w = game_weights(meta); w.sum()
    np.float64(2.0)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nfl.features.splits import q4_at
from nfl.models.metrics import EPS, Q4_MARKS

__all__ = [
    "Q4_LOG_KEY",
    "Q4_MARKS",
    "Q4Selection",
    "game_weights",
    "make_q4_callback",
    "q4_selection",
    "subset_inputs",
    "weighted_log_loss",
]

_GAME_KEYS = ("meta__game_id", "game_id")
_ROW = "__ga_row"


def _game_column(meta: pl.DataFrame, game_col: str | None = None) -> str:
    if game_col is not None:
        if game_col not in meta.columns:
            raise KeyError(f"frame has no {game_col!r}; columns={meta.columns}")
        return game_col
    for name in _GAME_KEYS:
        if name in meta.columns:
            return name
    raise KeyError(f"frame has no game id column; looked for {list(_GAME_KEYS)}")


# --------------------------------------------------------------------------- #
# Row weights
# --------------------------------------------------------------------------- #


def game_weights(meta: pl.DataFrame, *, game_col: str | None = None) -> np.ndarray:
    """``1 / plays-in-game`` for every row of ``meta``, in row order.

    The vector sums to the number of distinct games (up to float error), which
    is the invariant worth remembering: after weighting, the fit sees one unit
    of evidence per game rather than one per play.

    ``meta`` only needs the game key — pass the ``meta__`` frame, not the
    design matrix, since ``meta__game_id`` must never be a model input.
    """
    if meta.height == 0:
        return np.zeros(0, dtype=np.float64)
    col = _game_column(meta, game_col)
    counts = meta.select(pl.len().over(col).alias("__ga_n"))
    n = counts.get_column("__ga_n").cast(pl.Float64).to_numpy()
    if np.any(n <= 0):  # pragma: no cover - `count().over()` cannot produce this
        raise ValueError("a game came back with a non-positive play count")
    return (1.0 / n).astype(np.float64)


def weighted_log_loss(probs: object, y: object, weights: object | None = None) -> float:
    """Mean binary cross-entropy with optional per-row weights."""
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), EPS, 1.0 - EPS)
    t = np.asarray(y, dtype=np.float64).ravel()
    if p.shape != t.shape:
        raise ValueError(f"probs and y must have the same shape; got {p.shape} and {t.shape}")
    ll = -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))
    if weights is None:
        return float(np.mean(ll)) if ll.size else float("nan")
    w = np.asarray(weights, dtype=np.float64).ravel()
    total = float(np.sum(w))
    return float(np.sum(w * ll) / total) if total > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Q4 selection set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Q4Selection:
    """The rows model selection is decided on: three marks, one row per game each.

    Attributes:
        rows: positional indices into the frame ``q4_selection`` was given.
              A game appears once per mark it reaches, so the array is *not*
              unique — a row that is the last snapshot before both 600s and
              300s is listed twice, once under each mark.
        marks: for each entry of ``rows``, which mark it belongs to, as an index
               into ``mark_values``.
        mark_values: the marks themselves, in seconds remaining.
        weights: ``1 / n_rows_for_that_mark``. Sums to ``len(mark_values)``, so
                 a weighted mean over ``rows`` is the unweighted mean of the
                 three per-mark means however uneven the marks are.
    """

    rows: np.ndarray
    marks: np.ndarray
    mark_values: tuple[int, ...]
    weights: np.ndarray

    def __len__(self) -> int:
        return int(self.rows.size)

    @property
    def n_marks(self) -> int:
        return len(self.mark_values)

    def take(self, arr: object) -> np.ndarray:
        """``arr`` (indexed like the original frame) restricted to :attr:`rows`."""
        return np.asarray(arr)[self.rows]

    def per_mark_log_loss(self, probs: object, y: object) -> dict[int, float]:
        """Log-loss at each mark. ``probs``/``y`` are aligned with :attr:`rows`."""
        p = np.asarray(probs, dtype=np.float64).ravel()
        t = np.asarray(y, dtype=np.float64).ravel()
        return {
            int(mark): weighted_log_loss(p[self.marks == i], t[self.marks == i])
            for i, mark in enumerate(self.mark_values)
        }

    def mean_log_loss(self, probs: object, y: object) -> float:
        """The selection metric: mean val log-loss over the three Q4 marks.

        ``probs`` and ``y`` are aligned with :attr:`rows` (i.e. already
        subsetted). Equal to ``weighted_log_loss(probs, y, self.weights)``.
        """
        per_mark = self.per_mark_log_loss(probs, y)
        vals = [v for v in per_mark.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")


def q4_selection(
    meta: pl.DataFrame,
    *,
    marks: Sequence[int] = Q4_MARKS,
    tolerance: int | None = None,
) -> Q4Selection:
    """Build the Q4 selection set from a frame's ``meta__`` columns.

    ``meta`` needs ``meta__game_id``, ``meta__game_seconds_remaining`` and
    (ideally) ``meta__period``/``meta__play_id`` — i.e. the untouched
    validation frame, not the design matrix. The chosen row for a mark is the
    last snapshot *at or before* it, exactly as
    :func:`nfl.features.splits.q4_at` defines it for the reported metrics, so
    what the fit selects on and what the report prints are the same rows.
    """
    if meta.height == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Q4Selection(empty, empty, tuple(int(m) for m in marks), np.zeros(0))
    keyed = meta.with_row_index(_ROW)
    rows: list[np.ndarray] = []
    mark_ids: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for i, mark in enumerate(marks):
        picked = q4_at(keyed, int(mark), tolerance=tolerance)
        idx = picked.get_column(_ROW).cast(pl.Int64).to_numpy()
        if idx.size == 0:
            continue
        rows.append(idx)
        mark_ids.append(np.full(idx.size, i, dtype=np.int64))
        weights.append(np.full(idx.size, 1.0 / idx.size, dtype=np.float64))
    if not rows:
        empty = np.zeros(0, dtype=np.int64)
        return Q4Selection(empty, empty, tuple(int(m) for m in marks), np.zeros(0))
    return Q4Selection(
        rows=np.concatenate(rows),
        marks=np.concatenate(mark_ids),
        mark_values=tuple(int(m) for m in marks),
        weights=np.concatenate(weights),
    )


# --------------------------------------------------------------------------- #
# Keras callback
# --------------------------------------------------------------------------- #

#: What :func:`make_q4_callback` writes into the Keras ``logs`` dict.
Q4_LOG_KEY = "val_q4_log_loss"


def make_q4_callback(
    inputs: Any,
    y: np.ndarray,
    selection: Q4Selection,
    *,
    batch_size: int = 4096,
    key: str = Q4_LOG_KEY,
) -> Any:
    """A ``keras.callbacks.Callback`` that logs the Q4 selection metric each epoch.

    ``inputs`` is whatever the model's ``predict`` takes — an array for
    ``tf_mlp``, a ``{name: array}`` dict for ``tf_seq`` — *already restricted
    to* ``selection.rows``, and ``y`` likewise. Doing the subsetting up front
    means the per-epoch cost is one forward pass over ~10k rows rather than
    over the whole validation season.

    TensorFlow is imported inside this function, not at module scope: the CUDA
    preload in :func:`nfl.models.tf_common.configure_tensorflow` has to happen
    first, and importing this module must not force it.
    """
    import tensorflow as tf

    truth = np.asarray(y, dtype=np.float64).ravel()

    class Q4LogLoss(tf.keras.callbacks.Callback):  # type: ignore[misc, name-defined]
        """Writes the mean-over-marks Q4 log-loss into ``logs[key]``."""

        def __init__(self) -> None:
            super().__init__()
            self.history: list[float] = []

        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            probs = np.asarray(
                self.model.predict(inputs, batch_size=batch_size, verbose=0), dtype=np.float64
            ).ravel()
            value = selection.mean_log_loss(probs, truth)
            self.history.append(value)
            if logs is not None:
                logs[key] = value

    return Q4LogLoss()


def subset_inputs(inputs: Any, rows: np.ndarray) -> Any:
    """Restrict a Keras input (array or ``{name: array}``) to ``rows``."""
    if isinstance(inputs, dict):
        return {k: np.asarray(v)[rows] for k, v in inputs.items()}
    return np.asarray(inputs)[rows]
