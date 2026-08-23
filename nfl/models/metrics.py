"""Scoring a win-probability forecast: five numbers and the slices that matter.

Deliberately small and dependency-light — pure numpy in, floats out — because
these functions are called from inside training loops and from the fuller
evaluation module alike. Nothing here reads or writes anything.

**Ties.** The label is 1.0 / 0.0 / 0.5, and 0.5 is a real outcome (a tie), not
a missing value. ``log_loss`` and ``brier`` take it in their stride: a soft
label just splits the credit. ``accuracy`` gives a tie half credit whatever was
predicted, and ``auc`` — which has no meaning without two classes — drops tie
rows before ranking.

**Calibration.** ``ece`` is the standard 10-equal-width-bin expected
calibration error: the average |confidence - accuracy| gap, weighted by bin
population. ``reliability_table`` returns the same decomposition per bin, which
is what you actually plot.

**Slices.** :func:`slice_metrics` is the report the project is judged on:
overall, then the three late-Q4 marks (600s, 300s, 120s left), then two
"not collapsed" subsets.

``not_collapsed`` is keyed on ``reference_probs`` — the *floor model's*
probability sitting in ``[0.15, 0.85]``. Passing a reference is what makes two
models comparable on this slice: candidate and floor are then scored on
**identical rows**, so a difference in the number is a difference in skill
rather than a difference in population. Omit it and the reference is the
model's own probability, which is the right answer for the floor itself (its
own band *is* the reference band) and a self-selected sample for anything else.

``not_collapsed_own`` is always keyed on the model's own probability. It is
reported for information — "how does this model do on the rows *it* thinks are
live" — and is deliberately **not** what the promotion gate reads.

The market-implied version in :func:`nfl.features.splits.not_collapsed` needs a
market column that by ADR 0001 never exists on a feature frame, and is joined
in downstream; both bands here are the model-side stand-in until it lands.

    >>> import numpy as np
    >>> log_loss(np.array([0.9, 0.2]), np.array([1.0, 0.0])) < 0.2
    np.True_
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from nfl.features.splits import COLLAPSE_BOUNDS, q4_at

__all__ = [
    "COLLAPSE_BOUNDS",
    "EPS",
    "METRIC_NAMES",
    "NOT_COLLAPSED_OWN_SLICE",
    "NOT_COLLAPSED_SLICE",
    "Q4_MARKS",
    "accuracy",
    "auc",
    "brier",
    "ece",
    "log_loss",
    "metrics_dict",
    "reliability_table",
    "slice_metrics",
]

#: Probabilities are clipped into ``[EPS, 1 - EPS]`` before any log is taken.
EPS = 1e-15

#: Reported metrics, in the order they appear in every table.
METRIC_NAMES: tuple[str, ...] = ("log_loss", "brier", "accuracy", "auc", "ece")

#: Seconds remaining in regulation at the Q4 marks the project reports on.
Q4_MARKS: tuple[int, ...] = (600, 300, 120)

#: The gated not-collapsed slice: rows the *reference* model has not decided.
NOT_COLLAPSED_SLICE = "not_collapsed"

#: The informational variant: rows the model being scored has not decided.
NOT_COLLAPSED_OWN_SLICE = "not_collapsed_own"


def _prepared(probs: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    """``(p, y)`` as clean float arrays: same length, finite, p in ``(0, 1)``."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    if p.shape != t.shape:
        raise ValueError(f"probs and y must have the same shape; got {p.shape} and {t.shape}")
    keep = np.isfinite(p) & np.isfinite(t)
    return np.clip(p[keep], EPS, 1.0 - EPS), t[keep]


# --------------------------------------------------------------------------- #
# Scalar metrics
# --------------------------------------------------------------------------- #


def log_loss(probs: object, y: object) -> float:
    """Mean binary cross-entropy, in nats. Lower is better; 0.693 is a coin."""
    p, t = _prepared(probs, y)
    if p.size == 0:
        return float("nan")
    return float(-np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))


def brier(probs: object, y: object) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is a coin."""
    p, t = _prepared(probs, y)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - t) ** 2))


def accuracy(probs: object, y: object) -> float:
    """Share of correct side-picks, a tie counting half whatever was predicted."""
    p, t = _prepared(probs, y)
    if p.size == 0:
        return float("nan")
    picked_home = p > 0.5
    correct = np.where(t == 0.5, 0.5, np.where(picked_home, t, 1.0 - t))
    return float(np.mean(correct))


def auc(probs: object, y: object) -> float:
    """Rank AUC over decided games (ties dropped); ``nan`` if a class is absent.

    Computed from the Mann-Whitney statistic on ranks, so exact ties in the
    probability get the half-credit they deserve.
    """
    p, t = _prepared(probs, y)
    decided = t != 0.5
    p, t = p[decided], t[decided]
    n_pos = int(np.sum(t == 1.0))
    n_neg = int(np.sum(t == 0.0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    # Average the ranks inside each group of equal probabilities.
    sorted_p = p[order]
    start = 0
    for end in range(1, sorted_p.size + 1):
        if end == sorted_p.size or sorted_p[end] != sorted_p[start]:
            if end - start > 1:
                ranks[order[start:end]] = np.mean(ranks[order[start:end]])
            start = end
    rank_sum = float(np.sum(ranks[t == 1.0]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def reliability_table(probs: object, y: object, bins: int = 10) -> pl.DataFrame:
    """Per-bin ``(lo, hi, n, mean_prob, observed_rate, gap)`` over equal-width bins.

    Empty bins are kept, with ``n = 0`` and null statistics, so tables from two
    models always line up row for row.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1; got {bins}")
    p, t = _prepared(probs, y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # digitize with right=False puts p == 1.0 into a phantom bin; fold it back.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(np.sum(mask))
        mean_p = float(np.mean(p[mask])) if n else None
        rate = float(np.mean(t[mask])) if n else None
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "mean_prob": mean_p,
                "observed_rate": rate,
                "gap": None if n == 0 else abs(float(mean_p) - float(rate)),  # type: ignore[arg-type]
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "bin": pl.Int32,
            "lo": pl.Float64,
            "hi": pl.Float64,
            "n": pl.Int64,
            "mean_prob": pl.Float64,
            "observed_rate": pl.Float64,
            "gap": pl.Float64,
        },
    )


def ece(probs: object, y: object, bins: int = 10) -> float:
    """Expected calibration error over ``bins`` equal-width bins (default 10)."""
    table = reliability_table(probs, y, bins)
    n_total = int(table.get_column("n").sum())
    if n_total == 0:
        return float("nan")
    weighted = (
        table.get_column("n").cast(pl.Float64) * table.get_column("gap").fill_null(0.0)
    ).sum()
    return float(weighted) / n_total


def metrics_dict(probs: object, y: object, *, bins: int = 10) -> dict[str, float]:
    """All of :data:`METRIC_NAMES` at once, in order."""
    return {
        "log_loss": log_loss(probs, y),
        "brier": brier(probs, y),
        "accuracy": accuracy(probs, y),
        "auc": auc(probs, y),
        "ece": ece(probs, y, bins),
    }


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #

_PROB_COL = "_metrics__prob"
_LABEL_COL = "_metrics__y"
_REF_COL = "_metrics__ref"


def slice_metrics(
    df: pl.DataFrame,
    probs: object,
    y: object,
    *,
    reference_probs: object | None = None,
    bins: int = 10,
    q4_marks: Sequence[int] = Q4_MARKS,
    collapse_bounds: Sequence[float] = COLLAPSE_BOUNDS,
    tolerance: int | None = None,
) -> pl.DataFrame:
    """One row of metrics per slice: overall, each Q4 mark, and not-collapsed.

    ``df`` is the *evaluation* frame — it must still carry its ``meta__``
    columns, because that is what the Q4 slicing keys on — and ``probs``/``y``
    are positionally aligned with its rows. Returns a polars frame with
    ``slice``, ``n``, and :data:`METRIC_NAMES`.

    A Q4 row is "the last snapshot at or before the mark", one per game, via
    :func:`nfl.features.splits.q4_at`; a game that never reaches the mark
    contributes nothing.

    ``reference_probs`` is the floor model's probability on the *same rows*, and
    it is what :data:`NOT_COLLAPSED_SLICE` is keyed on. Two models given the
    same reference are scored on the same rows, which is the only way that
    comparison means anything (ADR 0003 §5). Omitted, the reference is ``probs``
    itself — right for the floor, self-selected for anything else.
    :data:`NOT_COLLAPSED_OWN_SLICE` always uses ``probs`` and is informational.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    if len(p) != df.height or len(t) != df.height:
        raise ValueError(
            f"probs/y must align with df rows; got {len(p)}/{len(t)} for {df.height} rows"
        )
    ref = p if reference_probs is None else np.asarray(reference_probs, dtype=np.float64).ravel()
    if len(ref) != df.height:
        raise ValueError(
            f"reference_probs must align with df rows; got {len(ref)} for {df.height} rows"
        )
    work = df.with_columns(
        pl.Series(_PROB_COL, p, dtype=pl.Float64),
        pl.Series(_LABEL_COL, t, dtype=pl.Float64),
        pl.Series(_REF_COL, ref, dtype=pl.Float64),
    ).filter(pl.col(_LABEL_COL).is_not_null() & pl.col(_PROB_COL).is_not_null())

    lo, hi = float(collapse_bounds[0]), float(collapse_bounds[1])
    parts: list[tuple[str, pl.DataFrame]] = [("overall", work)]
    for mark in q4_marks:
        parts.append((f"q4_{int(mark)}s", q4_at(work, int(mark), tolerance=tolerance)))
    parts.append(
        (
            NOT_COLLAPSED_SLICE,
            work.filter(pl.col(_REF_COL).is_between(lo, hi)),
        )
    )
    parts.append(
        (
            NOT_COLLAPSED_OWN_SLICE,
            work.filter(pl.col(_PROB_COL).is_between(lo, hi)),
        )
    )

    rows = []
    for name, part in parts:
        sp = part.get_column(_PROB_COL).to_numpy()
        sy = part.get_column(_LABEL_COL).to_numpy()
        rows.append({"slice": name, "n": part.height, **metrics_dict(sp, sy, bins=bins)})
    return pl.DataFrame(
        rows,
        schema={
            "slice": pl.String,
            "n": pl.Int64,
            **{m: pl.Float64 for m in METRIC_NAMES},
        },
    )
