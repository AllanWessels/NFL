"""``ScoreTimeLogistic`` — the floor every other model has to beat.

Nine raw columns and a logistic regression. Score margin, clock, whose ball it
is, timeouts, and where on the field: the things a fan reads off the scoreboard.
No team strength, no player data, no drive history. If a 300-tree GBDT over 775
features cannot beat *this*, the extra 766 features are not earning their keep
and the honest report says so.

Two modelling choices are worth naming.

**The interaction is the whole point.** A three-point lead is worth very
different things with 15 minutes left and with 30 seconds left, and a linear
model cannot discover that on its own. So the design matrix carries the margin
twice more: ``score_diff / sqrt(seconds + 1)`` — margin per unit of remaining
comeback capacity, which blows up correctly as the clock runs out — and
``score_diff * seconds/3600``, the plain product. Everything else enters
linearly.

**Ties are split, not dropped.** ``label__home_win`` is 0.5 for a tie, which
scikit-learn's ``LogisticRegression`` will not accept. Rather than throw the row
away, each tie is fitted as *both* outcomes at weight 0.5 — the exact
maximum-likelihood treatment of a soft label, and it keeps the row's features in
the fit.

Nulls are imputed with the training median and the medians are persisted, so a
prediction made a year later sees the same substitution the fit did.

    >>> model = ScoreTimeLogistic()                        # doctest: +SKIP
    >>> model.fit(X_train, y_train, X_val, y_val)          # doctest: +SKIP
    >>> model.predict_proba(X_val)[:3]                     # doctest: +SKIP
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.models.artifacts import assert_model_input_frame, model_dir, new_version, write_manifest
from nfl.models.base import FitResult
from nfl.models.data import ModelFrames, load_model_frames
from nfl.models.metrics import metrics_dict, slice_metrics

__all__ = [
    "MODEL_FILE",
    "NAME",
    "SOURCE_COLUMNS",
    "ScoreTimeLogistic",
    "design_terms",
    "train_naive",
]

log = logging.getLogger(__name__)

NAME = "naive_score_time"
MODEL_FILE = "model.joblib"

#: The only columns this model reads. Everything else in the dataset is ignored.
SOURCE_COLUMNS: tuple[str, ...] = (
    "bi__score_diff_home",
    "bi__game_seconds_remaining",
    "bi__posteam_is_home",
    "bi__home_timeouts_remaining",
    "bi__away_timeouts_remaining",
    "bi__timeout_diff_home",
    "bi__down",
    "bi__ydstogo",
    "bi__yardline_100",
)

#: The design matrix, in column order. Derived terms are built in
#: :func:`design_terms`; the names are what shows up in the coefficient table.
TERM_NAMES: tuple[str, ...] = (
    "score_diff",
    "score_diff_per_sqrt_time",
    "score_diff_x_time_frac",
    "time_frac",
    "log1p_seconds",
    "posteam_is_home",
    "timeout_diff_home",
    "timeouts_total",
    "down",
    "ydstogo",
    "yardline_100",
    "yardline_signed_home",
)

_FULL_GAME_SECONDS = 3600.0


def _col(X: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Float64, strict=False)


def design_terms(X: pl.DataFrame) -> pl.DataFrame:
    """Build the :data:`TERM_NAMES` design matrix from the raw ``bi__`` columns.

    Pure and null-preserving: imputation happens later, so the same function
    serves fit and predict.
    """
    missing = [c for c in SOURCE_COLUMNS if c not in X.columns]
    if missing:
        raise KeyError(f"{NAME} needs column(s) {missing}; got {len(X.columns)} columns")

    diff = _col(X, "bi__score_diff_home")
    secs = _col(X, "bi__game_seconds_remaining").clip(0.0, _FULL_GAME_SECONDS)
    posteam_home = _col(X, "bi__posteam_is_home")
    # -1 away ball / +1 home ball / null when nobody has it (kickoffs, timeouts).
    poss_sign = pl.when(posteam_home.is_null()).then(None).otherwise(posteam_home * 2.0 - 1.0)
    yardline = _col(X, "bi__yardline_100")

    return X.select(
        diff.alias("score_diff"),
        (diff / (secs + 1.0).sqrt()).alias("score_diff_per_sqrt_time"),
        (diff * secs / _FULL_GAME_SECONDS).alias("score_diff_x_time_frac"),
        (secs / _FULL_GAME_SECONDS).alias("time_frac"),
        (secs + 1.0).log().alias("log1p_seconds"),
        poss_sign.alias("posteam_is_home"),
        _col(X, "bi__timeout_diff_home").alias("timeout_diff_home"),
        (_col(X, "bi__home_timeouts_remaining") + _col(X, "bi__away_timeouts_remaining")).alias(
            "timeouts_total"
        ),
        _col(X, "bi__down").alias("down"),
        _col(X, "bi__ydstogo").alias("ydstogo"),
        yardline.alias("yardline_100"),
        # Field position from the home team's point of view: positive when the
        # home offence is close to scoring, negative when the away offence is.
        (poss_sign * (100.0 - yardline)).alias("yardline_signed_home"),
    )


def _soft_label_expansion(
    matrix: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn soft labels into ``(X, y_binary, weight)`` rows sklearn will accept.

    A decided game contributes one row at weight 1; a tie contributes two rows
    at weight 0.5 each, which is exactly its likelihood contribution.
    """
    decided = y != 0.5
    tied = ~decided
    if not np.any(tied):
        return matrix, y.astype(np.float64), np.ones(y.shape[0], dtype=np.float64)
    x_out = np.vstack([matrix[decided], matrix[tied], matrix[tied]])
    y_out = np.concatenate(
        [y[decided], np.ones(int(tied.sum())), np.zeros(int(tied.sum()))]
    ).astype(np.float64)
    w_out = np.concatenate([np.ones(int(decided.sum())), np.full(2 * int(tied.sum()), 0.5)]).astype(
        np.float64
    )
    return x_out, y_out, w_out


@dataclass
class ScoreTimeLogistic:
    """Logistic regression over score, clock and situation. The baseline floor.

    Attributes:
        C: inverse regularisation strength handed to ``LogisticRegression``.
        max_iter: lbfgs iteration cap.
        version: artifact version; assigned at :meth:`fit` if not given.
    """

    name: str = NAME
    C: float = 1.0
    max_iter: int = 1000
    version: str = ""
    _pipeline: Any = field(default=None, repr=False)
    _medians: dict[str, float] = field(default_factory=dict, repr=False)
    _scale: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)

    # -- internals --------------------------------------------------------- #

    def _matrix(self, X: pl.DataFrame, *, fit: bool = False) -> np.ndarray:
        terms = design_terms(X)
        if fit:
            self._medians = {
                c: float(v) if v is not None else 0.0
                for c, v in zip(
                    TERM_NAMES,
                    terms.select([pl.col(c).median() for c in TERM_NAMES]).row(0),
                    strict=True,
                )
            }
        if not self._medians:
            raise RuntimeError(f"{self.name} is not fitted")
        filled = terms.with_columns(
            [pl.col(c).fill_null(self._medians[c]).fill_nan(self._medians[c]) for c in TERM_NAMES]
        )
        matrix = filled.select(TERM_NAMES).to_numpy().astype(np.float64)
        if fit:
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            std[std < 1e-12] = 1.0
            self._scale = {
                c: (float(m), float(s)) for c, m, s in zip(TERM_NAMES, mean, std, strict=True)
            }
        mean = np.array([self._scale[c][0] for c in TERM_NAMES])
        std = np.array([self._scale[c][1] for c in TERM_NAMES])
        return (matrix - mean) / std

    # -- WinProbModel ------------------------------------------------------ #

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        """Fit on ``(X, y)``; report val metrics when a validation set is given."""
        from sklearn.linear_model import LogisticRegression

        assert_model_input_frame(X, context=f"{self.name}.fit")
        y = np.asarray(y, dtype=np.float64)
        matrix = self._matrix(X, fit=True)
        keep = np.isfinite(y)
        matrix, y = matrix[keep], y[keep]
        x_fit, y_fit, w_fit = _soft_label_expansion(matrix, y)

        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        clf.fit(x_fit, y_fit, sample_weight=w_fit)
        self._pipeline = clf
        self.version = self.version or new_version()

        metrics = {f"train_{k}": v for k, v in metrics_dict(self._raw_predict(matrix), y).items()}
        n_val = 0
        if X_val is not None and y_val is not None:
            yv = np.asarray(y_val, dtype=np.float64)
            n_val = int(np.isfinite(yv).sum())
            metrics.update(
                {f"val_{k}": v for k, v in metrics_dict(self.predict_proba(X_val), yv).items()}
            )
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(matrix.shape[0]),
            n_val=n_val,
            metrics=metrics,
            feature_names=list(SOURCE_COLUMNS),
            extra={"terms": list(TERM_NAMES), "coefficients": self.coefficients()},
        )

    def _raw_predict(self, matrix: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} is not fitted")
        probs: np.ndarray = self._pipeline.predict_proba(matrix)[:, 1]
        return probs

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """P(home win) for every row of ``X``."""
        assert_model_input_frame(X, context=f"{self.name}.predict_proba")
        return self._raw_predict(self._matrix(X))

    def coefficients(self) -> dict[str, float]:
        """Fitted coefficients by term name, plus ``intercept`` (standardised scale)."""
        if self._pipeline is None:
            return {}
        coefs = {c: float(v) for c, v in zip(TERM_NAMES, self._pipeline.coef_[0], strict=True)}
        coefs["intercept"] = float(self._pipeline.intercept_[0])
        return coefs

    # -- persistence ------------------------------------------------------- #

    def save(self, path: Path) -> None:
        """Write ``model.joblib`` into the directory ``path`` (created if absent)."""
        import joblib

        if self._pipeline is None:
            raise RuntimeError(f"{self.name} is not fitted; nothing to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "version": self.version,
                "C": self.C,
                "max_iter": self.max_iter,
                "pipeline": self._pipeline,
                "medians": self._medians,
                "scale": self._scale,
                "terms": list(TERM_NAMES),
                "source_columns": list(SOURCE_COLUMNS),
            },
            path / MODEL_FILE,
        )

    @classmethod
    def load(cls, path: Path) -> ScoreTimeLogistic:
        """Load from an artifact directory (or from the ``.joblib`` file itself)."""
        import joblib

        path = Path(path)
        target = path if path.is_file() else path / MODEL_FILE
        blob = joblib.load(target)
        model = cls(
            name=blob["name"],
            C=blob["C"],
            max_iter=blob["max_iter"],
            version=blob.get("version", ""),
        )
        model._pipeline = blob["pipeline"]
        model._medians = blob["medians"]
        model._scale = {k: tuple(v) for k, v in blob["scale"].items()}  # type: ignore[misc]
        return model


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #


def train_naive(
    seasons: str | int | Sequence[int] | None = None,
    *,
    frames: ModelFrames | None = None,
    root: Path | None = None,
    models_root: Path | None = None,
    stride: int = 1,
    save: bool = True,
) -> tuple[ScoreTimeLogistic, FitResult]:
    """Train :class:`ScoreTimeLogistic` on the assembled dataset and save it.

    Returns the fitted model and its :class:`~nfl.models.base.FitResult`; the
    result's ``extra`` carries the val and test slice tables as dictionaries.
    """
    data = frames if frames is not None else load_model_frames(seasons, root=root, stride=stride)
    x_train, y_train = data.as_xy("train")
    x_val, y_val = data.as_xy("val")

    model = ScoreTimeLogistic()
    result = model.fit(x_train, y_train, x_val, y_val)

    val_table = slice_metrics(data.val, model.predict_proba(x_val), y_val)
    result.extra["val_slices"] = val_table.to_dicts()
    if data.test.height:
        x_test, y_test = data.as_xy("test")
        test_table = slice_metrics(data.test, model.predict_proba(x_test), y_test)
        result.extra["test_slices"] = test_table.to_dicts()

    if save:
        from nfl.models.data import seasons_in

        target = model_dir(model.name, result.version, root=models_root)
        model.save(target)
        write_manifest(
            target,
            name=model.name,
            version=result.version,
            feature_names=SOURCE_COLUMNS,
            metrics=result.metrics,
            train_seasons=seasons_in(data.train),
            extra={
                "kind": "logistic_regression",
                "terms": list(TERM_NAMES),
                "coefficients": model.coefficients(),
                "stride": stride,
                "val_slices": result.extra.get("val_slices"),
                "test_slices": result.extra.get("test_slices"),
            },
        )
        log.info("wrote %s", target)
    return model, result
