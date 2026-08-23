"""Getting the assembled dataset into the shape ``fit`` wants.

One job: read ``data/processed/dataset``, drop the rows no model can learn from
(a game with no final score yet has a null label), cut it into train/val/test by
season, and hand back the ``(X, y)`` pair the :class:`~nfl.models.base.WinProbModel`
contract asks for — ``X`` is feature columns only, ``y`` is a float array with
0.5 for a tie.

``stride`` subsamples *plays*, not games: ``stride=3`` keeps every third row.
Consecutive plays in a game are nearly identical snapshots, so thinning them
costs far less signal than it saves wall time — which is what makes the
feature-group ablation in :mod:`nfl.models.gbdt` affordable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nfl.features.assemble import LABEL_COLUMN, feature_columns, read_dataset
from nfl.features.splits import DEFAULT_SPLIT, SeasonSplit, split_frames
from nfl.models.artifacts import assert_model_input_frame

__all__ = ["ModelFrames", "design_matrix", "labels", "load_model_frames", "seasons_in"]


@dataclass(frozen=True)
class ModelFrames:
    """The three season splits, still carrying their ``meta__`` columns.

    Keeping meta on the frame is deliberate: the Q4 slices in
    :func:`nfl.models.metrics.slice_metrics` need it. Strip it with
    :func:`design_matrix` at the last moment, right before ``fit``.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    features: list[str]
    split: SeasonSplit = DEFAULT_SPLIT

    def as_xy(self, which: str) -> tuple[pl.DataFrame, np.ndarray]:
        """``(X, y)`` for ``"train"`` / ``"val"`` / ``"test"``."""
        frame = getattr(self, which)
        return design_matrix(frame, self.features), labels(frame)


def design_matrix(df: pl.DataFrame, features: Sequence[str] | None = None) -> pl.DataFrame:
    """Feature columns only — no ``meta__``, no ``label__``, no market column."""
    cols = list(features) if features is not None else feature_columns(df)
    out = df.select(cols)
    assert_model_input_frame(out, context="design_matrix")
    return out


def labels(df: pl.DataFrame) -> np.ndarray:
    """``label__home_win`` as a float array: 1.0 home win, 0.0 loss, 0.5 tie."""
    if LABEL_COLUMN not in df.columns:
        raise KeyError(f"frame has no {LABEL_COLUMN}; it is not an assembled dataset")
    return df.get_column(LABEL_COLUMN).cast(pl.Float64).to_numpy()


def seasons_in(df: pl.DataFrame) -> list[int]:
    """Sorted distinct seasons present on a frame."""
    if "meta__season" not in df.columns:
        return []
    return sorted(int(s) for s in df.get_column("meta__season").drop_nulls().unique().to_list())


def load_model_frames(
    seasons: str | int | Sequence[int] | None = None,
    *,
    split: SeasonSplit = DEFAULT_SPLIT,
    root: Path | None = None,
    stride: int = 1,
    drop_string_features: bool = False,
) -> ModelFrames:
    """Read the dataset and cut it into train/val/test, labelled rows only.

    Args:
        seasons: restrict the read (``None`` reads everything on disk).
        split: which seasons go where; defaults to train <=2022 / val 2023 / test 2024+.
        root: dataset root override, for tests.
        stride: keep every ``stride``-th play (1 = all rows). Applied to the
            training frame only — val and test stay whole so the reported
            numbers are over the real population.
        drop_string_features: drop non-numeric feature columns outright, for
            models that cannot encode them.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    df = read_dataset(seasons, root=root)
    df = df.filter(pl.col(LABEL_COLUMN).is_not_null())
    feats = feature_columns(df)
    if drop_string_features:
        dtypes = dict(zip(df.columns, df.dtypes, strict=True))
        feats = [c for c in feats if dtypes[c] != pl.String]
    train, val, test = split_frames(df, split)
    if stride > 1:
        train = train.gather_every(stride)
    return ModelFrames(train=train, val=val, test=test, features=feats, split=split)
