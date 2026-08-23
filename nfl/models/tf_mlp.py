"""``tf_mlp`` — a tabular MLP over the assembled snapshot features.

The plain neural baseline: every feature column the assembler produces, median
imputed, standardised, with a null-indicator column beside each feature that is
ever missing, into two dense layers (256 / 128) with dropout and L2, and a
sigmoid on ``P(home wins)``.

The defaults are far more heavily regularised than a 620k-row table would
normally justify, and that is deliberate. Every play of a game carries that
game's outcome as its label and a nearly identical pre-game feature vector, so
620k rows are really ~3.4k independent examples wearing 1179 columns. Left
alone the net finds it easier to memorise which game each row belongs to than
to learn football: measured on this dataset, ``dropout=0.2, l2=1e-5`` drives
training log-loss to 0.011 by the third epoch while 2023 validation log-loss
climbs to 1.19 — worse than predicting 0.5 every time, on an AUC of 0.74 that
says the ranking was never the problem. The defaults here (dropout 0.5, L2
1e-3, unit Gaussian jitter on the standardised inputs) take the same
architecture to 0.485 validation log-loss at an ECE of 0.025. Turn the
regularisation down and the model does not get sharper, it gets confidently
wrong.

The interesting part is inference. Dropout is left switchable at predict time,
so the same weights give either a point estimate (``training=False``, the usual
thing) or a Monte-Carlo posterior: :meth:`MLPWinProb.predict_with_uncertainty`
runs ``n_samples`` stochastic forward passes and returns ``(mean, std)``. That
standard deviation is the model's own disagreement with itself, which is what
the downstream confidence score wants — a 0.60 that every dropout mask agrees
on is not the same bet as a 0.60 averaged over masks that ranged 0.3–0.9.

Artifacts follow the Phase 5 layout: ``data/models/tf_mlp/<version>/`` holding
``model.keras``, ``preprocessor.json``, ``config.json`` and ``manifest.json``.

    >>> result = train_tf_mlp()                      # doctest: +SKIP
    >>> mean, std = model.predict_with_uncertainty(X, n_samples=30)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.features.splits import DEFAULT_SPLIT, SeasonSplit
from nfl.models.base import FitResult
from nfl.models.feature_sets import feature_set_columns
from nfl.models.game_aware import (
    Q4_LOG_KEY,
    Q4Selection,
    game_weights,
    make_q4_callback,
    q4_selection,
    subset_inputs,
)
from nfl.models.tf_common import (
    PREPROCESSOR_FILE,
    TabularPreprocessor,
    TFRuntime,
    assert_model_input,
    binary_metrics,
    configure_tensorflow,
    labels_of,
    model_dir,
    new_version,
    read_modelling_frames,
    write_manifest,
)

__all__ = ["MLPWinProb", "train_tf_mlp"]

log = logging.getLogger(__name__)

MODEL_FILE = "model.keras"
CONFIG_FILE = "config.json"


class MLPWinProb:
    """Tabular MLP with persisted preprocessing and MC-dropout inference."""

    name = "tf_mlp"

    def __init__(
        self,
        *,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.5,
        input_noise: float = 1.0,
        input_dropout: float = 0.0,
        l2: float = 1e-3,
        learning_rate: float = 3e-4,
        batch_size: int = 4096,
        epochs: int = 80,
        patience: int = 8,
        seed: int = 17,
        selection_metric: str = "val",
        prefer_gpu: bool = True,
        mixed_precision: bool = True,
        verbose: int = 0,
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        if selection_metric not in {"val", "q4"}:
            raise ValueError(f"selection_metric must be 'val' or 'q4'; got {selection_metric!r}")
        self.selection_metric = str(selection_metric)
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout = float(dropout)
        self.input_noise = float(input_noise)
        self.input_dropout = float(input_dropout)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = int(seed)
        self.prefer_gpu = bool(prefer_gpu)
        self.mixed_precision = bool(mixed_precision)
        self.verbose = int(verbose)

        self.preprocessor = TabularPreprocessor()
        self.version: str = ""
        self.runtime: TFRuntime | None = None
        self.history: dict[str, list[float]] = {}
        self._model: Any = None

    # -- construction ------------------------------------------------------ #

    def _runtime(self) -> TFRuntime:
        if self.runtime is None:
            self.runtime = configure_tensorflow(
                prefer_gpu=self.prefer_gpu, mixed_precision=self.mixed_precision
            )
        return self.runtime

    def _build(self, n_features: int) -> Any:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        reg = tf.keras.regularizers.l2(self.l2) if self.l2 > 0 else None
        inputs = tf.keras.layers.Input(shape=(n_features,), name="features")
        h = inputs
        # Every row of a game shares that game's label, so ~620k rows are only
        # ~3.4k independent examples and a 1179-input net memorises which game
        # a row belongs to within one epoch. Jittering the inputs (and the
        # heavy dropout/L2 below) is what stops it: see the module docstring.
        if self.input_noise > 0:
            h = tf.keras.layers.GaussianNoise(self.input_noise, name="input_noise")(h)
        if self.input_dropout > 0:
            h = tf.keras.layers.Dropout(self.input_dropout, name="input_dropout")(h)
        for i, units in enumerate(self.hidden):
            h = tf.keras.layers.Dense(
                units, activation="relu", kernel_regularizer=reg, name=f"dense_{i}"
            )(h)
            if self.dropout > 0:
                h = tf.keras.layers.Dropout(self.dropout, name=f"dropout_{i}")(h)
        # float32 head: under a mixed_float16 policy the sigmoid and the loss
        # must stay in full precision or the log-loss quantises visibly.
        out = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="p_home")(h)
        model = tf.keras.Model(inputs, out, name="tf_mlp")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            # Keras folds the L2 penalty into `loss`, so early stopping on
            # `val_loss` would be stopping on "log-loss plus however big the
            # weights happen to be". This metric is the log-loss on its own,
            # and it is what the callbacks below watch.
            metrics=[tf.keras.metrics.BinaryCrossentropy(name="log_loss")],
        )
        return model

    # -- training ---------------------------------------------------------- #

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        *,
        sample_weight: np.ndarray | None = None,
        val_selection: Q4Selection | None = None,
    ) -> FitResult:
        """Fit on ``X``/``y``, early-stopping on the validation log-loss.

        ``sample_weight`` is the per-row weight (pass
        :func:`nfl.models.game_aware.game_weights` for game-aware training) and
        ``val_selection`` supplies the Q4 rows that ``selection_metric="q4"``
        watches instead of the overall validation log-loss.
        """
        assert_model_input(X, context=f"{self.name}.fit(X)")
        if X_val is not None:
            assert_model_input(X_val, context=f"{self.name}.fit(X_val)")
        runtime = self._runtime()
        import tensorflow as tf

        t_prep = time.perf_counter()
        train_x = self.preprocessor.fit_transform(X)
        # The sigmoid head is (n, 1) and `BinaryCrossentropy` as a *metric*
        # (unlike as a loss) refuses a rank mismatch, so shape y to match.
        train_y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        val_data = None
        if X_val is not None and y_val is not None and X_val.height:
            val_data = (
                self.preprocessor.transform(X_val),
                np.asarray(y_val, dtype=np.float32).reshape(-1, 1),
            )
        prep_seconds = time.perf_counter() - t_prep

        self._model = self._build(train_x.shape[1])
        monitor = "val_log_loss" if val_data is not None else "log_loss"
        callbacks: list[Any] = []
        # The Q4 callback has to run *before* EarlyStopping so the key it writes
        # is already in ``logs`` when the stopper reads it.
        if self.selection_metric == "q4":
            if val_data is None or val_selection is None or len(val_selection) == 0:
                raise ValueError(
                    "selection_metric='q4' needs a validation set and a non-empty val_selection"
                )
            callbacks.append(
                make_q4_callback(
                    subset_inputs(val_data[0], val_selection.rows),
                    val_data[1].ravel()[val_selection.rows],
                    val_selection,
                    batch_size=self.batch_size,
                )
            )
            monitor = Q4_LOG_KEY
        callbacks += [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=self.patience, restore_best_weights=True, mode="min"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=max(2, self.patience // 3),
                min_lr=1e-5,
                mode="min",
            ),
        ]
        t_fit = time.perf_counter()
        hist = self._model.fit(
            train_x,
            train_y,
            validation_data=val_data,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight, np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=self.verbose,
        )
        fit_seconds = time.perf_counter() - t_fit
        self.history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
        self.version = self.version or new_version()

        metrics: dict[str, float] = {}
        if val_data is not None:
            metrics = binary_metrics(
                val_data[1].astype(np.float64).ravel(), self._predict(val_data[0])
            )
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(train_x.shape[0]),
            n_val=int(val_data[0].shape[0]) if val_data is not None else 0,
            metrics=metrics,
            feature_names=list(self.preprocessor.feature_names),
            extra={
                "epochs_run": len(self.history.get("loss", [])),
                "fit_seconds": round(fit_seconds, 2),
                "preprocess_seconds": round(prep_seconds, 2),
                "n_inputs": int(train_x.shape[1]),
                "selection_metric": self.selection_metric,
                "sample_weighted": sample_weight is not None,
                "val_q4_log_loss": self.history.get(Q4_LOG_KEY, [float("nan")])[-1]
                if self.selection_metric == "q4"
                else None,
                "hyperparams": self._hyperparams(),
                "runtime": runtime.as_dict(),
            },
        )

    # -- inference --------------------------------------------------------- #

    def _predict(self, arr: np.ndarray) -> np.ndarray:
        """Deterministic forward pass (dropout off)."""
        out = self._model.predict(arr, batch_size=self.batch_size, verbose=0)
        return np.asarray(out, dtype=np.float64).ravel()

    def _stochastic(self, arr: np.ndarray) -> np.ndarray:
        """One MC-dropout forward pass (dropout on)."""
        chunks = [
            np.asarray(self._model(arr[i : i + self.batch_size], training=True), dtype=np.float64)
            for i in range(0, arr.shape[0], self.batch_size)
        ]
        return np.concatenate(chunks).ravel() if chunks else np.zeros(0)

    def _require_model(self, X: pl.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit() or load()")
        assert_model_input(X, context=f"{self.name}.predict")
        return self.preprocessor.transform(X)

    def predict_proba(self, X: pl.DataFrame, n_samples: int = 1) -> np.ndarray:
        """``P(home win)`` per row.

        ``n_samples=1`` (the default, and what the model contract calls for) is
        the ordinary deterministic pass. ``n_samples>1`` averages that many
        MC-dropout passes instead, which is the mean of
        :meth:`predict_with_uncertainty`.
        """
        arr = self._require_model(X)
        if n_samples <= 1:
            return self._predict(arr)
        total = np.zeros(arr.shape[0], dtype=np.float64)
        for _ in range(int(n_samples)):
            total += self._stochastic(arr)
        return total / float(n_samples)

    def predict_with_uncertainty(
        self, X: pl.DataFrame, n_samples: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)`` over ``n_samples`` MC-dropout passes.

        The spread comes from every stochastic layer the model has — dropout
        and, when it is enabled, the input jitter. With all of them off, every
        pass is identical and the standard deviation is exactly zero: correct,
        and a signal that the model was configured without a stochastic
        component rather than that it is certain.
        """
        n = max(2, int(n_samples))
        arr = self._require_model(X)
        total = np.zeros(arr.shape[0], dtype=np.float64)
        total_sq = np.zeros(arr.shape[0], dtype=np.float64)
        for _ in range(n):
            p = self._stochastic(arr)
            total += p
            total_sq += p * p
        mean = total / n
        var = np.maximum(total_sq / n - mean * mean, 0.0)
        return mean, np.sqrt(var)

    # -- persistence -------------------------------------------------------- #

    def _hyperparams(self) -> dict[str, Any]:
        return {
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "input_noise": self.input_noise,
            "input_dropout": self.input_dropout,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "selection_metric": self.selection_metric,
        }

    def save(self, path: Path) -> None:
        """Write ``model.keras`` + ``preprocessor.json`` + ``config.json`` into ``path``."""
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted; nothing to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(path / MODEL_FILE)
        self.preprocessor.save(path / PREPROCESSOR_FILE)
        (path / CONFIG_FILE).write_text(
            json.dumps(
                {"name": self.name, "version": self.version, "hyperparams": self._hyperparams()},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> MLPWinProb:
        """Rebuild a saved model, preprocessing included."""
        path = Path(path)
        config = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
        model = cls(**config.get("hyperparams", {}))
        model.version = str(config.get("version", ""))
        model.preprocessor = TabularPreprocessor.load(path / PREPROCESSOR_FILE)
        model._runtime()
        import tensorflow as tf

        model._model = tf.keras.models.load_model(path / MODEL_FILE)
        return model


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #


def train_tf_mlp(
    seasons: Sequence[int] | None = None,
    *,
    split: SeasonSplit = DEFAULT_SPLIT,
    root: Path | None = None,
    models_root: Path | None = None,
    save: bool = True,
    feature_set: str = "full",
    weight_by_game: bool = False,
    **kwargs: Any,
) -> tuple[MLPWinProb, FitResult]:
    """Train ``tf_mlp`` on the assembled dataset and persist the artifact.

    Seasons come from :mod:`nfl.features.splits` (train <= 2022, val 2023);
    ``kwargs`` are forwarded to :class:`MLPWinProb`. ``feature_set`` names the
    column selection (:mod:`nfl.models.feature_sets`) and ``weight_by_game``
    switches on the 1/plays-per-game row weights of ADR 0003.
    """
    train, val, _ = read_modelling_frames(seasons, split=split, root=root)
    if train.is_empty():
        raise ValueError("no training rows; run `nfl features build` first")
    features = feature_set_columns(train, feature_set)
    model = MLPWinProb(**kwargs)
    result = model.fit(
        train.select(features),
        labels_of(train),
        val.select(features),
        labels_of(val),
        sample_weight=game_weights(train.select("meta__game_id")) if weight_by_game else None,
        val_selection=q4_selection(val) if model.selection_metric == "q4" else None,
    )
    result.extra["feature_set"] = feature_set
    result.extra["weight_by_game"] = weight_by_game
    train_seasons = sorted(train.get_column("meta__season").unique().to_list())
    if save:
        directory = model_dir(model.name, result.version, root=models_root)
        model.save(directory)
        write_manifest(
            directory,
            name=model.name,
            version=result.version,
            train_seasons=train_seasons,
            feature_names=result.feature_names,
            metrics=result.metrics,
            extra={"model_file": MODEL_FILE, **result.extra},
        )
        result.extra["artifact_dir"] = str(directory)
        log.info("wrote %s", directory)
    result.extra["train_seasons"] = train_seasons
    return model, result


#: CLI entrypoint alias (``nfl train --model mlp``).
train_mlp = train_tf_mlp
