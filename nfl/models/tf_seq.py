"""``tf_seq`` — a sequence model over the play history inside the game.

``tf_mlp`` sees one snapshot at a time: the score, the clock, the aggregates.
What it cannot see is *shape* — a 10-point lead built on three straight
three-and-outs against a defence that just gave up two explosive plays reads
very differently from the same 10-point lead in a game nobody has moved in.
This model adds that: for every snapshot it also gets the last ``K = 32`` plays
as a compact per-play vector, encoded by a GRU, concatenated with the snapshot's
pre-game feature vector, and pushed through a shared head.

**Per-play vector** (:data:`PLAY_CHANNELS`, all home-signed so the model never
has to learn a possession convention): score differential, seconds remaining,
whether the offence is the home team, down, distance, field position, the EPA of
the play that just finished, the yards it gained, whether it was a turnover, and
the points it put on the board.

**Windowing is pure polars.** :func:`build_sequences` sorts by
``(game, play)``, derives the per-play deltas with one grouped shift each, then
materialises the ``K`` lags as ``K x F`` columns in a single ``select``. There is
no Python loop over plays anywhere; the only loops are over the 32 lags and the
10 channels, both of which are schema, not data. Lags reaching past the start of
a game are zeroed *and* flagged in a mask, so a window can never be filled from
the previous game — the leak this file most has to avoid.

**Masking.** Early snapshots have fewer than ``K`` plays behind them. The mask
rides along as an extra channel whose padded steps are all-zero, so a
``Masking`` layer makes the GRU skip them rather than learn from invented plays.

**Uncertainty.** :meth:`SeqWinProb.predict_with_uncertainty` runs ``n_samples``
forward passes with dropout and the pre-game jitter left on, and returns
``(mean, std)`` — the same MC-dropout head ``tf_mlp`` carries. That standard
deviation is the σ ADR 0003 §4 wanted as a confidence signal: a 0.62 that every
dropout mask agrees on is not the bet a 0.62 averaged over masks spanning
0.35–0.85 is.

    >>> model, result = train_tf_seq()                     # doctest: +SKIP
    >>> mean, std = model.predict_with_uncertainty(X, meta, n_samples=30)  # doctest: +SKIP
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

from nfl.features.assemble import feature_columns
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

__all__ = [
    "DEFAULT_K",
    "PLAY_CHANNELS",
    "SeqWinProb",
    "build_sequences",
    "play_channel_frame",
    "pregame_columns",
    "train_tf_seq",
]

log = logging.getLogger(__name__)

MODEL_FILE = "model.keras"
CONFIG_FILE = "config.json"

#: How many plays of history each snapshot carries.
DEFAULT_K = 32

#: The compact per-play vector, in channel order. Everything is signed from the
#: home team's point of view.
PLAY_CHANNELS: tuple[str, ...] = (
    "score_diff_home",
    "game_seconds_remaining",
    "posteam_is_home",
    "down",
    "ydstogo",
    "yardline_100",
    "play_epa_home",
    "yards_gained_home",
    "turnover_home",
    "score_change_home",
)

#: Where each channel comes from. ``None`` means "derived below".
_GAME_KEYS = ("meta__game_id", "game_id")
_ORDER_KEYS = ("ig__play_index", "meta__play_id", "play_id")

_POS = "__seq_pos"
_ROW = "__seq_row"


# --------------------------------------------------------------------------- #
# Column plumbing
# --------------------------------------------------------------------------- #


def pregame_columns(df: pl.DataFrame) -> list[str]:
    """Feature columns known before kickoff: everything that is not in-game.

    ``ig__``/``bi__`` describe the live state, and the sequence branch already
    carries that; the pre-game branch gets team form, player availability and
    game context.
    """
    return [
        c
        for c in feature_columns(df)
        if not c.startswith(("ig__", "bi__")) and not c.startswith("meta__")
    ]


def _first_present(df: pl.DataFrame, names: Sequence[str], what: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"frame has no {what} column; looked for {list(names)}")


#: The key a single serving row is windowed under. One row is one snapshot of
#: one game by construction, so the name is arbitrary — it only has to exist.
_SERVING_GAME_ID = "__serving__"


def _keyed(X: pl.DataFrame, meta: pl.DataFrame | None) -> pl.DataFrame:
    """``X`` with the game key and ordering column attached, row order intact.

    With no ``meta`` and no key already on the frame, a **single** row is keyed
    as one game: the serving path (``nfl predict``) builds exactly one snapshot,
    and a one-row window cannot reach into another game because there is no
    other row. More than one unkeyed row is refused rather than guessed at —
    inventing a shared key there would window across game boundaries, which is
    the leak this module exists to prevent.
    """
    if meta is None:
        if any(c in X.columns for c in _GAME_KEYS):
            return X
        if X.height != 1:
            raise KeyError(
                f"frame has no game id column (looked for {list(_GAME_KEYS)}) and "
                f"{X.height} rows; pass meta= so the window knows where each game "
                "starts. Only a single row can be keyed without it."
            )
        keyed = X.with_columns(pl.lit(_SERVING_GAME_ID).alias("meta__game_id"))
        if not any(c in keyed.columns for c in _ORDER_KEYS):
            keyed = keyed.with_columns(pl.lit(0.0).alias("meta__play_id"))
        return keyed
    if meta.height != X.height:
        raise ValueError(f"meta has {meta.height} rows, X has {X.height}")
    extra = [c for c in meta.columns if c not in X.columns]
    return pl.concat([X, meta.select(extra)], how="horizontal")


def _num(df: pl.DataFrame, name: str) -> pl.Expr:
    """``name`` as Float64, or a column of zeros when the dataset lacks it."""
    if name in df.columns:
        return pl.col(name).cast(pl.Float64, strict=False)
    return pl.lit(0.0, dtype=pl.Float64)


# --------------------------------------------------------------------------- #
# Per-play channels
# --------------------------------------------------------------------------- #


def play_channel_frame(
    df: pl.DataFrame, *, game_col: str | None = None, order_col: str | None = None
) -> pl.DataFrame:
    """One row per snapshot: ``__seq_row``, the game key and :data:`PLAY_CHANNELS`.

    Rows come back sorted by ``(game, play)``; ``__seq_row`` remembers where each
    one sat in the input so :func:`build_sequences` can put them back.

    The four "what just happened" channels are first differences of the running
    in-game totals over the game, which is exactly the outcome of the *previous*
    play — information the snapshot already has. The first play of a game has no
    predecessor, so its deltas are zero.
    """
    game_col = game_col or _first_present(df, _GAME_KEYS, "game id")
    order_col = order_col or _first_present(df, _ORDER_KEYS, "play ordering")

    frame = df.with_row_index(_ROW).sort([game_col, order_col, _ROW], maintain_order=True)

    def delta(name: str) -> pl.Expr:
        col = _num(frame, name)
        if name not in frame.columns:
            return col
        return (col - col.shift(1).over(game_col)).fill_null(0.0)

    channels = {
        "score_diff_home": _num(frame, "ig__score_diff_home"),
        "game_seconds_remaining": _num(frame, "ig__game_seconds_remaining"),
        "posteam_is_home": _num(frame, "ig__posteam_is_home"),
        "down": _num(frame, "ig__down"),
        "ydstogo": _num(frame, "ig__ydstogo"),
        "yardline_100": _num(frame, "ig__yardline_100"),
        "play_epa_home": delta("ig__epa_total_home") - delta("ig__epa_total_away"),
        "yards_gained_home": delta("ig__yards_home") - delta("ig__yards_away"),
        "turnover_home": (
            delta("ig__turnovers_committed_home") - delta("ig__turnovers_committed_away")
        ),
        "score_change_home": delta("ig__score_diff_home"),
    }
    return frame.select(
        pl.col(_ROW),
        pl.col(game_col).alias("__seq_game"),
        pl.int_range(pl.len()).over(game_col).alias(_POS),
        *[channels[name].fill_null(0.0).alias(name) for name in PLAY_CHANNELS],
    )


def channel_stats(chan: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-channel ``(mean, std)`` for standardising, zero-variance safe."""
    mean = chan.select([pl.col(c).mean().alias(c) for c in PLAY_CHANNELS]).row(0)
    std = chan.select([pl.col(c).std().alias(c) for c in PLAY_CHANNELS]).row(0)
    out: dict[str, tuple[float, float]] = {}
    for name, mu, sd in zip(PLAY_CHANNELS, mean, std, strict=True):
        s = 1.0 if sd is None or float(sd) <= 1e-12 else float(sd)
        out[name] = (0.0 if mu is None else float(mu), s)
    return out


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


def build_sequences(
    df: pl.DataFrame,
    *,
    k: int = DEFAULT_K,
    stats: dict[str, tuple[float, float]] | None = None,
    game_col: str | None = None,
    order_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    """``(sequences, mask, stats)`` for every row of ``df``, in input row order.

    ``sequences`` is ``(n, k, len(PLAY_CHANNELS))`` float32 with index ``k-1``
    holding the snapshot's own play and index ``0`` the oldest — so nothing at
    any index describes a play after the snapshot. ``mask`` is ``(n, k)``
    float32, ``1`` where the window holds a real play.

    ``stats`` standardises the channels; pass the training stats when
    transforming validation or test rows, or ``None`` to fit them here (the
    fitted stats come back as the third element either way).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    chan = play_channel_frame(df, game_col=game_col, order_col=order_col)
    stats = stats or channel_stats(chan)
    chan = chan.with_columns(
        [((pl.col(c) - stats[c][0]) / stats[c][1]).cast(pl.Float32).alias(c) for c in PLAY_CHANNELS]
    )

    # The frame is sorted by (game, play), so a plain ``shift(lag)`` is the
    # previous play of the same game whenever ``pos >= lag`` and belongs to the
    # *previous game* otherwise. ``pos >= lag`` is exactly the mask, so gating on
    # it both pads the window and makes cross-game bleed impossible.
    seq_exprs: list[pl.Expr] = []
    mask_exprs: list[pl.Expr] = []
    for lag in range(k - 1, -1, -1):
        valid = pl.col(_POS) >= lag
        mask_exprs.append(valid.cast(pl.Float32).alias(f"m__{lag}"))
        for name in PLAY_CHANNELS:
            seq_exprs.append(
                pl.when(valid)
                .then(pl.col(name).shift(lag))
                .otherwise(0.0)
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias(f"{name}__{lag}")
            )

    n = chan.height
    seq = chan.select(seq_exprs).to_numpy().reshape(n, k, len(PLAY_CHANNELS))
    mask = chan.select(mask_exprs).to_numpy()
    order = chan.get_column(_ROW).to_numpy()

    out_seq = np.empty_like(seq)
    out_mask = np.empty_like(mask)
    out_seq[order] = seq
    out_mask[order] = mask
    return (
        np.ascontiguousarray(out_seq, dtype=np.float32),
        np.ascontiguousarray(out_mask, dtype=np.float32),
        stats,
    )


def _model_order(seq: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip a chronological window to newest-first, which is what the GRU eats.

    :func:`build_sequences` returns windows oldest-first, so short histories are
    padded on the *left*. cuDNN's fused GRU kernel refuses any mask that is not
    contiguous ``True`` then contiguous ``False`` — right padding only — and
    Keras raises rather than silently falling back, so a left-padded mask fails
    outright on a GPU while passing on CPU. Reversing the window moves the
    padding to the tail; ``GRU(go_backwards=True)`` then reads it back in
    chronological order, and the fused kernel stays in play (measured at roughly
    twice the speed of ``use_cudnn=False`` on the 5080).

    The reversal lives here rather than in :func:`build_sequences` so the window
    the leakage tests inspect stays in the order a human would read it.
    """
    return seq[:, ::-1, :].copy(), mask[:, ::-1].copy()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class SeqWinProb:
    """GRU over the last ``k`` plays, joined with the pre-game feature vector."""

    name = "tf_seq"

    def __init__(
        self,
        *,
        k: int = DEFAULT_K,
        play_embed: int = 32,
        gru_units: int = 64,
        pregame_units: int = 128,
        head_units: int = 128,
        dropout: float = 0.5,
        pregame_noise: float = 1.0,
        l2: float = 1e-3,
        learning_rate: float = 3e-4,
        batch_size: int = 4096,
        epochs: int = 60,
        patience: int = 8,
        seed: int = 17,
        feature_set: str | None = None,
        selection_metric: str = "val",
        prefer_gpu: bool = True,
        mixed_precision: bool = True,
        verbose: int = 0,
    ) -> None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        if selection_metric not in {"val", "q4"}:
            raise ValueError(f"selection_metric must be 'val' or 'q4'; got {selection_metric!r}")
        # ``None`` keeps the Phase 5 behaviour (tabular branch = everything that
        # is not in-game); a name routes the branch through
        # :func:`nfl.models.feature_sets.feature_set_columns` instead, which is
        # how v1 hands the GRU's partner branch the in-play state as well.
        self.feature_set = None if feature_set is None else str(feature_set)
        self.selection_metric = str(selection_metric)
        self.k = int(k)
        self.play_embed = int(play_embed)
        self.gru_units = int(gru_units)
        self.pregame_units = int(pregame_units)
        self.head_units = int(head_units)
        self.dropout = float(dropout)
        self.pregame_noise = float(pregame_noise)
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
        self.channel_stats: dict[str, tuple[float, float]] = {}
        self.pregame_names: list[str] = []
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

    def _build(self, n_pregame: int) -> Any:
        import tensorflow as tf

        layers = tf.keras.layers
        tf.keras.utils.set_random_seed(self.seed)
        reg = tf.keras.regularizers.l2(self.l2) if self.l2 > 0 else None
        n_ch = len(PLAY_CHANNELS)

        plays = layers.Input(shape=(self.k, n_ch), name="plays")
        play_mask = layers.Input(shape=(self.k,), name="play_mask")
        mask3 = layers.Reshape((self.k, 1), name="mask3")(play_mask)
        # Zero the padded steps and hand the mask itself in as a channel, so a
        # padded step is all-zero and `Masking` can recognise it while a real
        # step never can be (its mask channel is 1).
        gated = layers.Multiply(name="gate")([plays, mask3])
        stacked = layers.Concatenate(axis=-1, name="with_mask")([gated, mask3])
        masked = layers.Masking(mask_value=0.0, name="masking")(stacked)
        embedded = layers.Dense(
            self.play_embed, activation="relu", kernel_regularizer=reg, name="play_embed"
        )(masked)
        # ``go_backwards`` walks the reversed window back into chronological
        # order, so the GRU still finishes on the snapshot's own play. See
        # :func:`_model_order` for why the window arrives reversed at all.
        encoded = layers.GRU(self.gru_units, go_backwards=True, name="gru")(embedded)

        pregame = layers.Input(shape=(n_pregame,), name="pregame")
        # Every row of a game carries the same ~530-wide pre-game vector and the
        # same label, so this branch is where the model would happily memorise
        # which game it is looking at. Jitter plus heavy dropout is the price of
        # letting it stay wide.
        p = (
            layers.GaussianNoise(self.pregame_noise, name="pregame_noise")(pregame)
            if self.pregame_noise > 0
            else pregame
        )
        p = layers.Dense(
            self.pregame_units, activation="relu", kernel_regularizer=reg, name="pregame_dense"
        )(p)
        if self.dropout > 0:
            p = layers.Dropout(self.dropout, name="pregame_dropout")(p)

        z = layers.Concatenate(name="join")([encoded, p])
        z = layers.Dense(self.head_units, activation="relu", kernel_regularizer=reg, name="head")(z)
        if self.dropout > 0:
            z = layers.Dropout(self.dropout, name="head_dropout")(z)
        out = layers.Dense(1, activation="sigmoid", dtype="float32", name="p_home")(z)

        model = tf.keras.Model([plays, play_mask, pregame], out, name="tf_seq")
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

    # -- feature preparation ------------------------------------------------ #

    def _inputs(
        self, X: pl.DataFrame, meta: pl.DataFrame | None, *, fit: bool
    ) -> dict[str, np.ndarray]:
        keyed = _keyed(X, meta)
        seq, mask, stats = build_sequences(
            keyed, k=self.k, stats=None if fit else self.channel_stats
        )
        seq, mask = _model_order(seq, mask)
        if fit:
            self.channel_stats = stats
            self.pregame_names = (
                pregame_columns(X)
                if self.feature_set is None
                else feature_set_columns(X, self.feature_set)
            )
            pre = self.preprocessor.fit_transform(X.select(self.pregame_names))
        else:
            pre = self.preprocessor.transform(X.select(self.pregame_names))
        return {"plays": seq, "play_mask": mask, "pregame": pre}

    # -- training ----------------------------------------------------------- #

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        *,
        meta: pl.DataFrame | None = None,
        meta_val: pl.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
        val_selection: Q4Selection | None = None,
    ) -> FitResult:
        """Fit on ``X``/``y``; ``meta`` supplies ``meta__game_id`` for windowing.

        ``X`` is features only, per the model contract, so the game key has to
        arrive separately — unless the frame happens to still carry it, which is
        what the ``meta=None`` path allows for ad-hoc use.

        ``sample_weight`` and ``val_selection`` are the ADR 0003 game-aware
        knobs: per-row weights (see
        :func:`nfl.models.game_aware.game_weights`) and the Q4 rows that
        ``selection_metric="q4"`` early-stops on.
        """
        assert_model_input(X, context=f"{self.name}.fit(X)")
        if X_val is not None:
            assert_model_input(X_val, context=f"{self.name}.fit(X_val)")
        runtime = self._runtime()
        import tensorflow as tf

        t_prep = time.perf_counter()
        train_in = self._inputs(X, meta, fit=True)
        # The sigmoid head is (n, 1) and `BinaryCrossentropy` as a *metric*
        # (unlike as a loss) refuses a rank mismatch, so shape y to match.
        train_y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        val_data = None
        if X_val is not None and y_val is not None and X_val.height:
            val_data = (
                self._inputs(X_val, meta_val, fit=False),
                np.asarray(y_val, dtype=np.float32).reshape(-1, 1),
            )
        prep_seconds = time.perf_counter() - t_prep

        self._model = self._build(train_in["pregame"].shape[1])
        monitor = "val_log_loss" if val_data is not None else "log_loss"
        callbacks: list[Any] = []
        # Ordered before EarlyStopping so the key it writes is already in
        # ``logs`` when the stopper reads it.
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
            train_in,
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
        self.history = {kk: [float(v) for v in vs] for kk, vs in hist.history.items()}
        self.version = self.version or new_version()

        metrics: dict[str, float] = {}
        if val_data is not None:
            metrics = binary_metrics(
                val_data[1].astype(np.float64).ravel(), self._predict(val_data[0])
            )
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(train_y.shape[0]),
            n_val=int(val_data[1].shape[0]) if val_data is not None else 0,
            metrics=metrics,
            feature_names=list(PLAY_CHANNELS) + list(self.preprocessor.feature_names),
            extra={
                "epochs_run": len(self.history.get("loss", [])),
                "fit_seconds": round(fit_seconds, 2),
                "preprocess_seconds": round(prep_seconds, 2),
                "k": self.k,
                "n_play_channels": len(PLAY_CHANNELS),
                "n_pregame_inputs": int(train_in["pregame"].shape[1]),
                "feature_set": self.feature_set,
                "selection_metric": self.selection_metric,
                "sample_weighted": sample_weight is not None,
                "val_q4_log_loss": self.history.get(Q4_LOG_KEY, [float("nan")])[-1]
                if self.selection_metric == "q4"
                else None,
                "hyperparams": self._hyperparams(),
                "runtime": runtime.as_dict(),
            },
        )

    # -- inference ----------------------------------------------------------- #

    def _predict(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        out = self._model.predict(inputs, batch_size=self.batch_size, verbose=0)
        return np.asarray(out, dtype=np.float64).ravel()

    def _stochastic(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """One MC-dropout forward pass — every stochastic layer left switched on.

        ``training=True`` is what keeps ``pregame_dropout``, ``head_dropout``
        and the ``pregame_noise`` jitter live at inference. Called in batches
        because the whole validation split at once does not fit in 16 GB.
        """
        n = next(iter(inputs.values())).shape[0]
        chunks = [
            np.asarray(
                self._model(
                    {k: v[i : i + self.batch_size] for k, v in inputs.items()}, training=True
                ),
                dtype=np.float64,
            )
            for i in range(0, n, self.batch_size)
        ]
        return np.concatenate(chunks).ravel() if chunks else np.zeros(0)

    def _require_inputs(self, X: pl.DataFrame, meta: pl.DataFrame | None) -> dict[str, np.ndarray]:
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit() or load()")
        assert_model_input(X, context=f"{self.name}.predict")
        return self._inputs(X, meta, fit=False)

    def predict_proba(
        self, X: pl.DataFrame, meta: pl.DataFrame | None = None, n_samples: int = 1
    ) -> np.ndarray:
        """``P(home win)`` per row of ``X``.

        ``n_samples=1`` (the model contract's shape) is the ordinary
        deterministic pass. Above that it averages that many MC-dropout passes,
        which is exactly the mean :meth:`predict_with_uncertainty` returns.
        """
        inputs = self._require_inputs(X, meta)
        if n_samples <= 1:
            return self._predict(inputs)
        total = np.zeros(next(iter(inputs.values())).shape[0], dtype=np.float64)
        for _ in range(int(n_samples)):
            total += self._stochastic(inputs)
        return total / float(n_samples)

    def predict_with_uncertainty(
        self, X: pl.DataFrame, meta: pl.DataFrame | None = None, n_samples: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)`` over ``n_samples`` MC-dropout passes.

        The GRU's own disagreement with itself, and the second confidence signal
        ADR 0003 §4 asked for. The windowing is done once and the same tensors
        are re-run, so ``n_samples`` costs forward passes and not feature work.

        With ``dropout=0`` and ``pregame_noise=0`` nothing is stochastic, every
        pass is identical and the standard deviation is exactly zero — correct,
        and a signal that the model was built without a stochastic component
        rather than that it is certain.
        """
        n = max(2, int(n_samples))
        inputs = self._require_inputs(X, meta)
        rows = next(iter(inputs.values())).shape[0]
        total = np.zeros(rows, dtype=np.float64)
        total_sq = np.zeros(rows, dtype=np.float64)
        for _ in range(n):
            p = self._stochastic(inputs)
            total += p
            total_sq += p * p
        mean = total / n
        var = np.maximum(total_sq / n - mean * mean, 0.0)
        return mean, np.sqrt(var)

    # -- persistence ---------------------------------------------------------- #

    def _hyperparams(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "play_embed": self.play_embed,
            "gru_units": self.gru_units,
            "pregame_units": self.pregame_units,
            "head_units": self.head_units,
            "dropout": self.dropout,
            "pregame_noise": self.pregame_noise,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "feature_set": self.feature_set,
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
                {
                    "name": self.name,
                    "version": self.version,
                    "hyperparams": self._hyperparams(),
                    "pregame_names": self.pregame_names,
                    "channel_stats": {k: list(v) for k, v in self.channel_stats.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SeqWinProb:
        """Rebuild a saved model, windowing statistics included."""
        path = Path(path)
        config = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
        model = cls(**config.get("hyperparams", {}))
        model.version = str(config.get("version", ""))
        model.pregame_names = list(config.get("pregame_names", []))
        model.channel_stats = {
            k: (float(v[0]), float(v[1])) for k, v in config.get("channel_stats", {}).items()
        }
        model.preprocessor = TabularPreprocessor.load(path / PREPROCESSOR_FILE)
        model._runtime()
        import tensorflow as tf

        model._model = tf.keras.models.load_model(path / MODEL_FILE)
        return model


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #

#: The columns the windowing needs but that must never be model inputs.
SEQ_META_COLUMNS = ("meta__game_id", "meta__play_id")


def train_tf_seq(
    seasons: Sequence[int] | None = None,
    *,
    split: SeasonSplit = DEFAULT_SPLIT,
    root: Path | None = None,
    models_root: Path | None = None,
    save: bool = True,
    weight_by_game: bool = False,
    **kwargs: Any,
) -> tuple[SeqWinProb, FitResult]:
    """Train ``tf_seq`` on the assembled dataset and persist the artifact.

    ``kwargs`` go to :class:`SeqWinProb` — including ``feature_set`` and
    ``selection_metric``. ``X`` always keeps every feature column so the
    per-play channels can be derived; the *tabular branch* is what the feature
    set narrows.
    """
    train, val, _ = read_modelling_frames(seasons, split=split, root=root)
    if train.is_empty():
        raise ValueError("no training rows; run `nfl features build` first")
    features = feature_columns(train)
    meta_cols = [c for c in SEQ_META_COLUMNS if c in train.columns]
    model = SeqWinProb(**kwargs)
    result = model.fit(
        train.select(features),
        labels_of(train),
        val.select(features),
        labels_of(val),
        meta=train.select(meta_cols),
        meta_val=val.select(meta_cols),
        sample_weight=game_weights(train.select("meta__game_id")) if weight_by_game else None,
        val_selection=q4_selection(val) if model.selection_metric == "q4" else None,
    )
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
            extra={
                "model_file": MODEL_FILE,
                "play_channels": list(PLAY_CHANNELS),
                **result.extra,
            },
        )
        result.extra["artifact_dir"] = str(directory)
        log.info("wrote %s", directory)
    result.extra["train_seasons"] = train_seasons
    return model, result


#: CLI entrypoint alias (``nfl train --model seq``).
train_seq = train_tf_seq
