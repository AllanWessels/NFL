"""Metrics: the five numbers, the reliability decomposition, and the slices."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nfl.models.metrics import (
    METRIC_NAMES,
    accuracy,
    auc,
    brier,
    ece,
    log_loss,
    metrics_dict,
    reliability_table,
    slice_metrics,
)


def test_perfect_forecast_scores_zero_loss() -> None:
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert log_loss(p, y) == pytest.approx(0.0, abs=1e-9)
    assert brier(p, y) == pytest.approx(0.0)
    assert accuracy(p, y) == pytest.approx(1.0)
    assert auc(p, y) == pytest.approx(1.0)
    assert ece(p, y) == pytest.approx(0.0)


def test_coin_flip_is_the_reference_point() -> None:
    y = np.array([1.0, 0.0] * 50)
    p = np.full(100, 0.5)
    assert log_loss(p, y) == pytest.approx(math.log(2), abs=1e-9)
    assert brier(p, y) == pytest.approx(0.25)


def test_log_loss_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.random(500) < p).astype(float)
    from sklearn.metrics import log_loss as sk_log_loss

    assert log_loss(p, y) == pytest.approx(sk_log_loss(y, p), rel=1e-9)


def test_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(0.0, 1.0, 500)
    y = (rng.random(500) < p).astype(float)
    from sklearn.metrics import roc_auc_score

    assert auc(p, y) == pytest.approx(float(roc_auc_score(y, p)), rel=1e-9)


def test_ties_are_half_credit_not_a_crash() -> None:
    """A 0.5 label is a tie, a real outcome — it must not be dropped silently."""
    p = np.array([0.9, 0.1, 0.5])
    y = np.array([1.0, 0.0, 0.5])
    assert accuracy(p, y) == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    # auc has no meaning for a tie, so it ranks the two decided rows only.
    assert auc(p, y) == pytest.approx(1.0)
    assert np.isfinite(log_loss(p, y))
    assert brier(p, y) == pytest.approx((0.01 + 0.01 + 0.0) / 3)


def test_auc_is_nan_when_one_class_is_missing() -> None:
    assert math.isnan(auc(np.array([0.6, 0.7]), np.array([1.0, 1.0])))


def test_reliability_table_has_one_row_per_bin_and_sums_to_n() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0.0, 1.0, 1000)
    y = (rng.random(1000) < p).astype(float)
    table = reliability_table(p, y, bins=10)
    assert table.height == 10
    assert int(table.get_column("n").sum()) == 1000
    assert table.get_column("lo").to_list()[0] == 0.0
    assert table.get_column("hi").to_list()[-1] == 1.0


def test_ece_is_zero_for_a_perfectly_calibrated_forecast() -> None:
    """Each bin's mean probability equals its observed rate by construction."""
    p = np.concatenate([np.full(1000, 0.25), np.full(1000, 0.75)])
    y = np.concatenate([np.repeat([1.0, 0.0], [250, 750]), np.repeat([1.0, 0.0], [750, 250])])
    assert ece(p, y, bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_reports_the_size_of_a_deliberate_miscalibration() -> None:
    """Always says 0.9, actually happens 0.6 of the time -> ECE 0.3."""
    p = np.full(1000, 0.9)
    y = np.repeat([1.0, 0.0], [600, 400])
    assert ece(p, y, bins=10) == pytest.approx(0.3, abs=1e-9)


def test_probability_of_one_lands_in_the_top_bin() -> None:
    table = reliability_table(np.array([1.0, 1.0]), np.array([1.0, 1.0]), bins=10)
    assert int(table.get_column("n").to_list()[-1]) == 2


def test_metrics_dict_returns_every_named_metric() -> None:
    out = metrics_dict(np.array([0.6, 0.4]), np.array([1.0, 0.0]))
    assert list(out) == list(METRIC_NAMES)


def _eval_frame(n_games: int = 6) -> pl.DataFrame:
    """A tiny assembled-shaped frame: one game per row-block, clock counting down."""
    clocks = [900, 700, 610, 590, 400, 290, 200, 110, 60, 0]
    rows = []
    for g in range(n_games):
        for i, clock in enumerate(clocks):
            rows.append(
                {
                    "meta__game_id": f"2024_0{g + 1}_AA_BB",
                    "meta__play_id": float(i),
                    "meta__game_seconds_remaining": clock,
                    "meta__period": 4,
                    "meta__season": 2024,
                }
            )
    return pl.DataFrame(rows)


def test_slice_metrics_reports_overall_q4_and_not_collapsed() -> None:
    df = _eval_frame()
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.02, 0.98, df.height)
    y = (rng.random(df.height) < probs).astype(float)
    table = slice_metrics(df, probs, y)
    assert table.get_column("slice").to_list() == [
        "overall",
        "q4_600s",
        "q4_300s",
        "q4_120s",
        "not_collapsed",
    ]
    for metric in METRIC_NAMES:
        assert metric in table.columns
    overall = table.filter(pl.col("slice") == "overall").row(0, named=True)
    assert overall["n"] == df.height


def test_q4_slices_take_one_row_per_game_at_or_before_the_mark() -> None:
    df = _eval_frame(n_games=4)
    probs = np.full(df.height, 0.5)
    y = np.zeros(df.height)
    table = slice_metrics(df, probs, y)
    counts = dict(zip(table.get_column("slice"), table.get_column("n"), strict=True))
    assert counts["q4_600s"] == 4
    assert counts["q4_300s"] == 4
    assert counts["q4_120s"] == 4


def test_not_collapsed_drops_rows_the_model_has_already_decided() -> None:
    df = _eval_frame(n_games=2)
    probs = np.where(np.arange(df.height) % 2 == 0, 0.99, 0.5)
    y = np.ones(df.height)
    table = slice_metrics(df, probs, y)
    counts = dict(zip(table.get_column("slice"), table.get_column("n"), strict=True))
    assert counts["not_collapsed"] == int((probs == 0.5).sum())
    assert counts["overall"] == df.height


def test_slice_metrics_rejects_misaligned_probabilities() -> None:
    df = _eval_frame(n_games=1)
    with pytest.raises(ValueError, match="align"):
        slice_metrics(df, np.zeros(3), np.zeros(3))


def test_metrics_ignore_rows_with_a_null_label() -> None:
    p = np.array([0.9, 0.1, 0.5])
    y = np.array([1.0, 0.0, np.nan])
    assert log_loss(p, y) == pytest.approx(log_loss(p[:2], y[:2]))
