"""Shared plumbing for the TensorFlow win-probability models.

Three things live here so ``tf_mlp`` and ``tf_seq`` cannot drift apart:

**Runtime setup.** :func:`configure_tensorflow` does the CUDA preload that
``CLAUDE.md`` demands *before* ``import tensorflow``, turns on memory growth and
mixed precision when a GPU is present, and then deliberately runs a throwaway
fit to pay the RTX 5080's PTX JIT cost up front. TF 2.21 ships no kernels for
compute capability 12.0a, so the first dense/GRU op of a process compiles from
PTX; timing it separately keeps that cost out of the reported training time.

**Tabular preprocessing.** :class:`TabularPreprocessor` is the persisted
median-impute + standardise + null-indicator transform the model contract asks
for. Fitted statistics are JSON, not pickle, so an artifact stays readable and
loadable across library versions.

**Artifacts.** :func:`write_manifest` writes the Phase 5 ``manifest.json``
alongside the ``.keras`` file, and :func:`new_version` mints the UTC-timestamp
version string that names the directory.

Metric helpers are a thin shim: if a branch has ``nfl.models.metrics`` it is
used, otherwise equivalent minimal implementations here are.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.config import MODELS_DIR
from nfl.data.market_blocklist import assert_no_market_columns
from nfl.features.assemble import LABEL_COLUMN, LABEL_PREFIX, META_PREFIX, read_dataset
from nfl.features.splits import SeasonSplit, Splits, split_frames

__all__ = [
    "NON_GENERALISING_SUFFIXES",
    "PREPROCESSOR_FILE",
    "TFRuntime",
    "TabularPreprocessor",
    "assert_model_input",
    "binary_metrics",
    "configure_tensorflow",
    "git_sha",
    "labels_of",
    "model_dir",
    "new_version",
    "read_modelling_frames",
    "write_manifest",
]

log = logging.getLogger(__name__)

PREPROCESSOR_FILE = "preprocessor.json"
MANIFEST_FILE = "manifest.json"

#: Calendar identifiers. ``home_tm__season`` is 2010..2022 in training and 2023+
#: at inference: a value the network has never seen, on a column whose only
#: content is *which* season this is. Standardised it lands off the end of the
#: training range and the first layer extrapolates; kept in, it also lets a wide
#: net memorise which game a row belongs to. Neither is a thing we want, so the
#: preprocessor drops it. (Suffix is deliberately ``__season`` with two
#: underscores: ``home_tm__win_pct_season`` is a real rolling statistic and
#: stays.)
NON_GENERALISING_SUFFIXES: tuple[str, ...] = ("__season",)

#: Numeric polars dtypes a feature column may have. Everything else (the
#: handful of ``*_player_id`` / ``franchise`` / ``venue_id`` string columns) is
#: identity, not signal, and is dropped by the preprocessor rather than hashed.
_NUMERIC = (
    pl.Boolean,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)


# --------------------------------------------------------------------------- #
# Input hygiene
# --------------------------------------------------------------------------- #


def assert_model_input(X: pl.DataFrame, *, context: str = "model input") -> None:
    """Fail loudly if ``X`` carries market, label or meta columns.

    The three hard rules of the project in one call: ADR 0001 (no market
    inputs), CLAUDE.md rule 2 (the label never rides along as a feature) and the
    Phase 5 contract (``X`` is features only).
    """
    assert_no_market_columns(X, context=context)
    leaked = [c for c in X.columns if c.startswith((LABEL_PREFIX, META_PREFIX))]
    if leaked:
        raise AssertionError(f"{context}: label__/meta__ columns must not be features: {leaked}")


def labels_of(df: pl.DataFrame) -> np.ndarray:
    """``label__home_win`` as a float64 array (ties are 0.5)."""
    return df.get_column(LABEL_COLUMN).cast(pl.Float64).to_numpy().astype(np.float64)


def read_modelling_frames(
    seasons: Sequence[int] | None = None,
    *,
    split: SeasonSplit | None = None,
    root: Path | None = None,
) -> Splits:
    """``(train, val, test)`` assembled rows with a known outcome.

    Rows whose game has not been played yet carry a null label and are dropped
    here rather than in every caller.
    """
    df = read_dataset(list(seasons) if seasons else None, root=root)
    df = df.filter(pl.col(LABEL_COLUMN).is_not_null())
    return split_frames(df, split)


# --------------------------------------------------------------------------- #
# TensorFlow runtime
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TFRuntime:
    """What :func:`configure_tensorflow` found and how long the JIT took."""

    gpu: bool
    devices: list[str]
    mixed_precision: bool
    import_seconds: float
    warmup_seconds: float
    tf_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "devices": self.devices,
            "mixed_precision": self.mixed_precision,
            "import_seconds": round(self.import_seconds, 2),
            "warmup_seconds": round(self.warmup_seconds, 2),
            "tf_version": self.tf_version,
        }


_RUNTIME: TFRuntime | None = None


def configure_tensorflow(
    *, prefer_gpu: bool = True, mixed_precision: bool = True, warm_up: bool = True
) -> TFRuntime:
    """Import TensorFlow correctly and pay the PTX JIT cost up front.

    Cached: the first call in a process configures, later calls return the same
    :class:`TFRuntime`. ``prefer_gpu=False`` hides the GPU from TensorFlow via
    ``CUDA_VISIBLE_DEVICES``, which only works if it happens before the import —
    hence the caching.
    """
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    from nfl.gpu import preload_cuda_libraries

    preload_cuda_libraries()
    if not prefer_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    t0 = time.perf_counter()
    import tensorflow as tf

    import_seconds = time.perf_counter() - t0

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Already-initialised contexts refuse the change; nothing to do then.
        with contextlib.suppress(RuntimeError):
            tf.config.experimental.set_memory_growth(gpu, True)
    use_mixed = bool(gpus) and mixed_precision
    if use_mixed:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    warmup_seconds = _warm_up(tf) if (warm_up and gpus) else 0.0
    _RUNTIME = TFRuntime(
        gpu=bool(gpus),
        devices=[g.name for g in gpus],
        mixed_precision=use_mixed,
        import_seconds=import_seconds,
        warmup_seconds=warmup_seconds,
        tf_version=tf.__version__,
    )
    log.info("tensorflow runtime: %s", _RUNTIME.as_dict())
    return _RUNTIME


def _warm_up(tf: Any) -> float:
    """Force the PTX JIT for the kernels both models need. Returns seconds."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    x = rng.random((256, 8), dtype=np.float32)
    s = rng.random((256, 4, 3), dtype=np.float32)
    y = (rng.random(256) > 0.5).astype(np.float32)

    dense = tf.keras.Sequential(
        [
            tf.keras.layers.Input((8,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )
    dense.compile(optimizer="adam", loss="binary_crossentropy")
    dense.fit(x, y, epochs=1, batch_size=128, verbose=0)

    si = tf.keras.layers.Input((4, 3))
    h = tf.keras.layers.Conv1D(4, 2, padding="causal", activation="relu")(si)
    h = tf.keras.layers.GRU(8)(h)
    seq = tf.keras.Model(si, tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(h))
    seq.compile(optimizer="adam", loss="binary_crossentropy")
    seq.fit(s, y, epochs=1, batch_size=128, verbose=0)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TabularPreprocessor:
    """Median imputation + standardisation + null-indicator columns.

    Fitted on the training frame and persisted with the model, so inference
    reproduces training exactly even when a later frame has different nulls.

    Columns that are non-numeric, entirely null in training, or a calendar
    identifier (:data:`NON_GENERALISING_SUFFIXES`) are dropped and recorded in
    :attr:`dropped_names` — a constant column carries no signal, a null median
    is not a number, and a season number is neither.

    Standardised values are clipped to +/-:attr:`clip` sigma. Season-held-out
    validation puts rows outside the training range on almost every rolling
    statistic, and an unclipped z-score of 30 walks a ReLU stack straight off
    the map; clipping costs nothing on in-range rows and bounds the damage on
    the rest.
    """

    feature_names: list[str] = field(default_factory=list)
    dropped_names: list[str] = field(default_factory=list)
    indicator_names: list[str] = field(default_factory=list)
    medians: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    clip: float = 5.0

    # -- fitting ---------------------------------------------------------- #

    def fit(self, X: pl.DataFrame) -> TabularPreprocessor:
        """Learn medians, means, stds and which columns need a null indicator."""
        numeric = [
            c
            for c in X.columns
            if X.schema[c] in _NUMERIC and not c.endswith(NON_GENERALISING_SUFFIXES)
        ]
        dropped = [c for c in X.columns if c not in numeric]

        nulls = X.select([pl.col(c).null_count().alias(c) for c in numeric]).row(0)
        n = X.height
        all_null = {c for c, k in zip(numeric, nulls, strict=True) if k == n}
        kept = [c for c in numeric if c not in all_null]
        dropped += sorted(all_null)

        med = X.select([pl.col(c).cast(pl.Float64).median().alias(c) for c in kept]).row(0)
        medians = [0.0 if m is None else float(m) for m in med]

        filled = X.select(
            [
                pl.col(c).cast(pl.Float64).fill_null(m).alias(c)
                for c, m in zip(kept, medians, strict=True)
            ]
        )
        mean = filled.select([pl.col(c).mean().alias(c) for c in kept]).row(0)
        std = filled.select([pl.col(c).std().alias(c) for c in kept]).row(0)
        means = [0.0 if m is None else float(m) for m in mean]
        # A zero (or null) standard deviation means a constant column; dividing
        # by 1.0 leaves it at exactly 0.0 rather than producing inf/nan.
        stds = [1.0 if s is None or float(s) <= 1e-12 else float(s) for s in std]

        self.feature_names = kept
        self.dropped_names = dropped
        self.indicator_names = [
            c for c, k in zip(numeric, nulls, strict=True) if c in set(kept) and k > 0
        ]
        self.medians, self.means, self.stds = medians, means, stds
        return self

    # -- applying --------------------------------------------------------- #

    @property
    def output_names(self) -> list[str]:
        """Column names of :meth:`transform`'s output, in order."""
        return self.feature_names + [f"{c}__isnull" for c in self.indicator_names]

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names) + len(self.indicator_names)

    def transform(self, X: pl.DataFrame) -> np.ndarray:
        """``(n, n_features_out)`` float32: standardised values then indicators."""
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise KeyError(f"frame is missing {len(missing)} fitted feature columns: {missing[:5]}")
        lo, hi = -abs(self.clip), abs(self.clip)
        exprs = [
            ((pl.col(c).cast(pl.Float64).fill_null(m) - mu) / sd)
            .clip(lo, hi)
            # polars keeps null and NaN apart, so `fill_null` above does not
            # catch a float NaN that came in from the source data; 0.0 is the
            # training mean, i.e. the same "no information" value imputation
            # lands on.
            .fill_nan(0.0)
            .cast(pl.Float32)
            .alias(c)
            for c, m, mu, sd in zip(
                self.feature_names, self.medians, self.means, self.stds, strict=True
            )
        ]
        exprs += [
            pl.col(c).is_null().cast(pl.Float32).alias(f"{c}__isnull") for c in self.indicator_names
        ]
        return np.ascontiguousarray(X.select(exprs).to_numpy(), dtype=np.float32)

    def fit_transform(self, X: pl.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "dropped_names": self.dropped_names,
            "indicator_names": self.indicator_names,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TabularPreprocessor:
        return cls(**payload)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TabularPreprocessor:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #


def new_version(now: datetime | None = None) -> str:
    """UTC timestamp version string, ``YYYYMMDDTHHMMSS`` (Phase 5 convention)."""
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")


def git_sha(short: bool = False) -> str | None:
    """The current commit, or ``None`` outside a git checkout."""
    repo = Path(__file__).resolve().parents[2]
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def model_dir(name: str, version: str, *, root: Path | None = None) -> Path:
    """``data/models/<name>/<version>``."""
    return (root or MODELS_DIR) / name / version


def write_manifest(
    directory: Path,
    *,
    name: str,
    version: str,
    train_seasons: Sequence[int],
    feature_names: Sequence[str],
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` next to the model file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "name": name,
        "version": version,
        "git_sha": git_sha(),
        "train_seasons": list(train_seasons),
        "features": list(feature_names),
        "n_features": len(feature_names),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    path = directory / MANIFEST_FILE
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Metrics (delegating to nfl.models.metrics when a branch provides it)
# --------------------------------------------------------------------------- #


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, dtype=np.float64) - _clip(p)) ** 2))


def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((_clip(p) >= 0.5) == (np.asarray(y, dtype=np.float64) >= 0.5)))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = _clip(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        total += (sel.sum() / y.size) * abs(y[sel].mean() - p[sel].mean())
    return float(total)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC; ties in ``y`` at 0.5 count as half-positive."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    pos, neg = y >= 0.75, y <= 0.25
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    # Average ranks within tied probability groups.
    sorted_p = p[order]
    start = 0
    for i in range(1, p.size + 1):
        if i == p.size or sorted_p[i] != sorted_p[start]:
            ranks[order[start:i]] = (start + i + 1) / 2.0
            start = i
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


try:  # pragma: no cover - depends on which branch supplied nfl/models/metrics.py
    from nfl.models.metrics import binary_metrics  # type: ignore[no-redef]
except ImportError:

    def binary_metrics(y: np.ndarray, p: np.ndarray, *, bins: int = 10) -> dict[str, float]:
        """log_loss / brier / accuracy / ece / auc, in the Phase 5 naming."""
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        if y.size == 0:
            return dict.fromkeys(("log_loss", "brier", "accuracy", "ece", "auc"), float("nan"))
        return {
            "log_loss": _log_loss(y, p),
            "brier": _brier(y, p),
            "accuracy": _accuracy(y, p),
            "ece": _ece(y, p, bins=bins),
            "auc": _auc(y, p),
        }
