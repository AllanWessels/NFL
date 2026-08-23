"""Tests for the ADR 0003 game-aware training pieces.

Three things have to hold, and each is checked against an independent
calculation rather than against the implementation restated:

* ``game_weights`` sums to the number of games — the invariant that makes the
  weighting "one unit of evidence per game" rather than per play;
* ``q4_selection`` picks the same rows the reported metrics do, one per game
  per mark, and never a row from after the mark;
* the Keras callback writes a number ``EarlyStopping`` can act on, and that
  number is the mean of the three per-mark log-losses.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nfl.features.splits import q4_at
from nfl.models.game_aware import (
    Q4_LOG_KEY,
    game_weights,
    make_q4_callback,
    q4_selection,
    subset_inputs,
    weighted_log_loss,
)


def synthetic_meta(n_games: int = 6, plays_per_game: tuple[int, ...] = (9, 7, 11)) -> pl.DataFrame:
    """A meta frame with uneven game lengths and a clock that runs down through Q4."""
    rows = []
    for g in range(n_games):
        n = plays_per_game[g % len(plays_per_game)]
        # Clock from 880s down to ~40s: every game crosses 600, 300 and 120.
        for i in range(n):
            rows.append(
                {
                    "meta__game_id": f"g{g}",
                    "meta__play_id": i,
                    "meta__period": 4,
                    "meta__game_seconds_remaining": int(880 - i * (840 // max(n - 1, 1))),
                }
            )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# game_weights
# --------------------------------------------------------------------------- #


def test_game_weights_sums_to_the_number_of_games():
    meta = synthetic_meta(n_games=6)
    w = game_weights(meta)
    assert w.shape == (meta.height,)
    assert math.isclose(float(w.sum()), meta.get_column("meta__game_id").n_unique())


def test_game_weights_is_one_over_plays_in_game():
    meta = pl.DataFrame({"meta__game_id": ["a", "a", "a", "b"]})
    assert game_weights(meta).tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])


def test_game_weights_follows_row_order_not_sorted_order():
    meta = pl.DataFrame({"meta__game_id": ["b", "a", "a"]})
    assert game_weights(meta).tolist() == pytest.approx([1.0, 0.5, 0.5])


def test_game_weights_accepts_a_bare_game_id_column():
    assert game_weights(pl.DataFrame({"game_id": ["x", "x"]})).tolist() == pytest.approx([0.5, 0.5])


def test_game_weights_on_an_empty_frame_is_empty():
    assert game_weights(pl.DataFrame({"meta__game_id": []})).shape == (0,)


def test_game_weights_without_a_game_column_raises():
    with pytest.raises(KeyError, match="game id"):
        game_weights(pl.DataFrame({"meta__season": [2024]}))


def test_weighted_log_loss_matches_the_unweighted_mean_when_weights_are_equal():
    p = np.array([0.7, 0.2, 0.9])
    y = np.array([1.0, 0.0, 1.0])
    assert weighted_log_loss(p, y) == pytest.approx(weighted_log_loss(p, y, np.full(3, 4.0)))


# --------------------------------------------------------------------------- #
# q4_selection
# --------------------------------------------------------------------------- #


def test_q4_selection_picks_one_row_per_game_per_mark():
    meta = synthetic_meta(n_games=6)
    sel = q4_selection(meta)
    assert sel.n_marks == 3
    assert len(sel) == 6 * 3
    for i in range(3):
        assert np.unique(sel.rows[sel.marks == i]).size == 6


def test_q4_selection_weights_sum_to_the_number_of_marks():
    sel = q4_selection(synthetic_meta())
    assert float(sel.weights.sum()) == pytest.approx(sel.n_marks)


def test_q4_selection_rows_agree_with_q4_at():
    meta = synthetic_meta(n_games=4)
    sel = q4_selection(meta)
    for i, mark in enumerate(sel.mark_values):
        expected = q4_at(meta.with_row_index("__r"), mark).get_column("__r").to_list()
        assert sorted(sel.rows[sel.marks == i].tolist()) == sorted(expected)


def test_q4_selection_never_picks_a_row_from_after_the_mark():
    meta = synthetic_meta(n_games=5)
    clock = meta.get_column("meta__game_seconds_remaining").to_numpy()
    sel = q4_selection(meta)
    for i, mark in enumerate(sel.mark_values):
        assert np.all(clock[sel.rows[sel.marks == i]] >= mark)


def test_q4_selection_on_an_empty_frame_is_empty():
    sel = q4_selection(
        pl.DataFrame(
            schema={
                "meta__game_id": pl.String,
                "meta__game_seconds_remaining": pl.Int32,
                "meta__period": pl.Int32,
            }
        )
    )
    assert len(sel) == 0


def test_mean_log_loss_is_the_mean_of_the_per_mark_log_losses():
    meta = synthetic_meta(n_games=5)
    sel = q4_selection(meta)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=len(sel))
    y = (rng.random(len(sel)) > 0.5).astype(float)
    per_mark = sel.per_mark_log_loss(p, y)
    assert sel.mean_log_loss(p, y) == pytest.approx(float(np.mean(list(per_mark.values()))))
    # And equals the weighted pooled loss, which is what XGBoost early-stops on.
    assert sel.mean_log_loss(p, y) == pytest.approx(weighted_log_loss(p, y, sel.weights))


def test_take_and_subset_inputs_line_up():
    meta = synthetic_meta(n_games=3)
    sel = q4_selection(meta)
    arr = np.arange(meta.height, dtype=float)
    assert sel.take(arr).tolist() == arr[sel.rows].tolist()
    d = subset_inputs({"a": arr, "b": arr * 2}, sel.rows)
    assert d["a"].tolist() == arr[sel.rows].tolist()
    assert d["b"].tolist() == (arr * 2)[sel.rows].tolist()


# --------------------------------------------------------------------------- #
# The Keras callback
# --------------------------------------------------------------------------- #


def test_q4_callback_logs_the_selection_metric_each_epoch():
    """A three-epoch fit on synthetic data: the key lands in ``logs``/``history``."""
    from nfl.models.tf_common import configure_tensorflow

    configure_tensorflow(prefer_gpu=False, mixed_precision=False, warm_up=False)
    import tensorflow as tf

    meta = synthetic_meta(n_games=8)
    sel = q4_selection(meta)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(meta.height, 4)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=0.3, size=meta.height) > 0).astype(np.float32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")

    cb = make_q4_callback(x[sel.rows], y[sel.rows], sel, batch_size=64)
    hist = model.fit(
        x,
        y.reshape(-1, 1),
        epochs=3,
        batch_size=16,
        verbose=0,
        callbacks=[
            cb,
            tf.keras.callbacks.EarlyStopping(monitor=Q4_LOG_KEY, patience=5, mode="min"),
        ],
    )

    assert Q4_LOG_KEY in hist.history
    assert len(hist.history[Q4_LOG_KEY]) == 3
    assert all(np.isfinite(v) and v > 0 for v in hist.history[Q4_LOG_KEY])
    assert cb.history == pytest.approx(hist.history[Q4_LOG_KEY])

    # The logged value is the selection metric on those rows, not something else.
    probs = np.asarray(model.predict(x[sel.rows], verbose=0)).ravel()
    assert cb.history[-1] == pytest.approx(
        sel.mean_log_loss(probs, y[sel.rows].astype(np.float64)), rel=1e-4
    )


def test_q4_callback_can_early_stop_a_fit():
    """``EarlyStopping`` monitoring the Q4 key actually fires (no silent no-op)."""
    from nfl.models.tf_common import configure_tensorflow

    configure_tensorflow(prefer_gpu=False, mixed_precision=False, warm_up=False)
    import tensorflow as tf

    meta = synthetic_meta(n_games=6)
    sel = q4_selection(meta)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(meta.height, 3)).astype(np.float32)
    y = np.ones(meta.height, dtype=np.float32)

    model = tf.keras.Sequential(
        [tf.keras.layers.Input((3,)), tf.keras.layers.Dense(1, activation="sigmoid")]
    )
    # A huge learning rate makes the metric get worse, so patience=1 must bite
    # well before the 40th epoch.
    model.compile(optimizer=tf.keras.optimizers.Adam(5.0), loss="binary_crossentropy")
    hist = model.fit(
        x,
        y.reshape(-1, 1),
        epochs=40,
        batch_size=8,
        verbose=0,
        callbacks=[
            make_q4_callback(x[sel.rows], y[sel.rows], sel, batch_size=64),
            tf.keras.callbacks.EarlyStopping(monitor=Q4_LOG_KEY, patience=1, mode="min"),
        ],
    )
    assert len(hist.history[Q4_LOG_KEY]) < 40
