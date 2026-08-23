"""Tests for the two TensorFlow models.

Everything here runs on CPU with a handful of epochs over a synthetic frame —
the point is that the plumbing is right (preprocessing persists, artifacts round
trip, windows never reach into the future), not that a 200-row fit learns
anything.

The metric helpers at the top are the "minimal log_loss/brier/ece" the Phase 5
conventions allow a test file to carry when a branch has no
``nfl.models.metrics`` yet: they exist so
:func:`nfl.models.tf_common.binary_metrics` is checked against an independent
implementation rather than against itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl.models.base import FitResult, WinProbModel
from nfl.models.tf_common import TabularPreprocessor, binary_metrics, new_version, write_manifest
from nfl.models.tf_mlp import MLPWinProb
from nfl.models.tf_seq import (
    DEFAULT_K,
    PLAY_CHANNELS,
    SeqWinProb,
    _model_order,
    build_sequences,
    play_channel_frame,
    pregame_columns,
)

# --------------------------------------------------------------------------- #
# Minimal reference metrics (test-local; see module docstring)
# --------------------------------------------------------------------------- #


def ref_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def ref_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def ref_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(total)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #

#: Plays per game in the synthetic frame; the short game is deliberately
#: shorter than K so the padding/masking path is always exercised.
GAME_LENGTHS = {"G_long": 40, "G_short": 5, "G_mid": 20}


def synthetic_frame(seed: int = 0, lengths: dict[str, int] | None = None) -> pl.DataFrame:
    """An assembled-dataset lookalike: meta + label + ``ig__``/``ctx__`` features."""
    rng = np.random.default_rng(seed)
    lengths = lengths or GAME_LENGTHS
    rows: list[dict[str, object]] = []
    for game, n_plays in lengths.items():
        season = 2020 + (hash(game) % 3)
        home_win = float(rng.integers(0, 2))
        base = float(rng.normal(0, 7))
        for i in range(n_plays):
            rows.append(
                {
                    "meta__season": season,
                    "meta__game_id": game,
                    "meta__play_id": float(i * 10),
                    "meta__game_seconds_remaining": float(3600 - i * 12),
                    "meta__period": 1 + min(3, i // 10),
                    "label__home_win": home_win,
                    "ig__play_index": i,
                    "ig__score_diff_home": base + i * (1.0 if home_win else -1.0),
                    "ig__game_seconds_remaining": float(3600 - i * 12),
                    "ig__posteam_is_home": float(i % 2),
                    "ig__down": float(1 + i % 4),
                    "ig__ydstogo": float(1 + i % 10),
                    "ig__yardline_100": float(20 + i % 60),
                    "ig__epa_total_home": float(i) * 0.3,
                    "ig__epa_total_away": float(i) * 0.2,
                    "ig__yards_home": float(i) * 5.0,
                    "ig__yards_away": float(i) * 4.0,
                    "ig__turnovers_committed_home": float(i // 7),
                    "ig__turnovers_committed_away": float(i // 11),
                    "ctx__temp_f": None if i % 13 == 0 else 55.0 + rng.normal(),
                    "ctx__div_game": float(hash(game) % 2),
                    "ctx__home_rest_days": 7.0,
                    "ctx__constant": 1.0,
                    "ctx__all_null": None,
                }
            )
    return pl.DataFrame(rows, strict=False)


def features_of(df: pl.DataFrame) -> pl.DataFrame:
    from nfl.features.assemble import feature_columns

    return df.select(feature_columns(df))


def labels_of(df: pl.DataFrame) -> np.ndarray:
    return df.get_column("label__home_win").to_numpy().astype(np.float64)


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return synthetic_frame()


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_preprocessor_imputes_standardises_and_flags_nulls(frame: pl.DataFrame) -> None:
    X = features_of(frame)
    pre = TabularPreprocessor()
    out = pre.fit_transform(X)

    assert out.dtype == np.float32
    assert out.shape == (X.height, pre.n_features_out)
    assert np.isfinite(out).all(), "imputation must leave no NaN/inf behind"

    # The all-null and non-numeric columns are dropped; the constant column
    # survives but standardises to exactly zero rather than dividing by zero.
    assert "ctx__all_null" in pre.dropped_names
    assert "ctx__constant" in pre.feature_names
    const = out[:, pre.feature_names.index("ctx__constant")]
    assert np.allclose(const, 0.0)

    # Only the column that actually has nulls gets an indicator, and it is 1
    # exactly on the null rows.
    assert pre.indicator_names == ["ctx__temp_f"]
    flag = out[:, pre.output_names.index("ctx__temp_f__isnull")]
    expected = X.get_column("ctx__temp_f").is_null().to_numpy().astype(np.float32)
    assert np.array_equal(flag, expected)

    # Standardised columns have (near) zero mean and unit variance on train.
    temp = out[:, pre.feature_names.index("ctx__temp_f")]
    assert abs(float(temp.mean())) < 1e-4


def test_preprocessor_clips_out_of_range_values_and_drops_season_ids() -> None:
    """Season-held-out rows land off the end of the training range; clip them."""
    train = pl.DataFrame(
        {
            "ctx__temp_f": [50.0, 51.0, 52.0, 53.0],
            "home_tm__season": [2010.0, 2011.0, 2012.0, 2013.0],
            "home_tm__win_pct_season": [0.1, 0.2, 0.3, 0.4],
        }
    )
    pre = TabularPreprocessor(clip=3.0).fit(train)
    # The bare season identifier goes; the rolling season statistic stays.
    assert "home_tm__season" in pre.dropped_names
    assert "home_tm__win_pct_season" in pre.feature_names

    later = train.with_columns(pl.Series("ctx__temp_f", [50.0, 51.0, 52.0, 5000.0]))
    out = pre.transform(later)
    assert out.max() <= 3.0 and out.min() >= -3.0


def test_preprocessor_round_trips_through_json(tmp_path: Path, frame: pl.DataFrame) -> None:
    X = features_of(frame)
    pre = TabularPreprocessor().fit(X)
    path = tmp_path / "preprocessor.json"
    pre.save(path)
    back = TabularPreprocessor.load(path)
    assert back.to_dict() == pre.to_dict()
    assert np.array_equal(back.transform(X), pre.transform(X))


def test_preprocessor_rejects_a_frame_missing_fitted_columns(frame: pl.DataFrame) -> None:
    X = features_of(frame)
    pre = TabularPreprocessor().fit(X)
    with pytest.raises(KeyError, match="missing"):
        pre.transform(X.drop("ctx__temp_f"))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_binary_metrics_matches_an_independent_implementation() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(500) > 0.4).astype(np.float64)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.15, 500), 0.01, 0.99)
    got = binary_metrics(y, p)
    assert got["log_loss"] == pytest.approx(ref_log_loss(y, p), rel=1e-9)
    assert got["brier"] == pytest.approx(ref_brier(y, p), rel=1e-9)
    assert got["ece"] == pytest.approx(ref_ece(y, p), abs=1e-9)
    assert 0.5 < got["auc"] <= 1.0
    assert 0.0 <= got["accuracy"] <= 1.0


def test_binary_metrics_on_an_empty_slice_is_all_nan() -> None:
    got = binary_metrics(np.zeros(0), np.zeros(0))
    assert all(math.isnan(v) for v in got.values())


# --------------------------------------------------------------------------- #
# Sequence windows
# --------------------------------------------------------------------------- #

RAW_STATS = dict.fromkeys(PLAY_CHANNELS, (0.0, 1.0))
SCORE_DIFF = PLAY_CHANNELS.index("score_diff_home")


def test_play_channels_are_deltas_of_the_running_totals(frame: pl.DataFrame) -> None:
    chan = play_channel_frame(frame)
    assert chan.height == frame.height
    assert list(chan.columns[-len(PLAY_CHANNELS) :]) == list(PLAY_CHANNELS)

    long_game = chan.filter(pl.col("__seq_game") == "G_long").sort("__seq_pos")
    # ig__epa_total_home rises 0.3/play and away 0.2/play, so the home-signed
    # play EPA is a flat 0.1 -- except on the first play, which has no
    # predecessor and is therefore zero.
    epa = long_game.get_column("play_epa_home").to_numpy()
    assert epa[0] == pytest.approx(0.0)
    assert np.allclose(epa[1:], 0.1)
    yards = long_game.get_column("yards_gained_home").to_numpy()
    assert yards[0] == pytest.approx(0.0)
    assert np.allclose(yards[1:], 1.0)


def test_window_never_contains_a_future_play_and_is_masked_at_the_start() -> None:
    """The load-bearing leakage test for the sequence branch."""
    df = synthetic_frame()
    # A per-row marker that is unique across the whole frame, so a value showing
    # up at the wrong lag (or from the wrong game) is unmistakable.
    marker = np.arange(df.height, dtype=np.float64) + 1.0
    df = df.with_columns(pl.Series("ig__score_diff_home", marker))

    seq, mask, _ = build_sequences(df, k=DEFAULT_K, stats=RAW_STATS)
    assert seq.shape == (df.height, DEFAULT_K, len(PLAY_CHANNELS))
    assert mask.shape == (df.height, DEFAULT_K)

    pos = df.get_column("ig__play_index").to_numpy()
    for i in range(df.height):
        for lag in range(DEFAULT_K):
            slot = DEFAULT_K - 1 - lag
            value = seq[i, slot, SCORE_DIFF]
            if lag <= pos[i]:
                assert mask[i, slot] == 1.0
                assert value == pytest.approx(marker[i - lag]), (
                    f"row {i} lag {lag} pulled the wrong play"
                )
            else:
                # Past the start of the game: masked, zeroed, and specifically
                # *not* the tail of the previous game.
                assert mask[i, slot] == 0.0
                assert value == 0.0

    # The newest slot is always the snapshot's own play, never a later one.
    assert np.array_equal(seq[:, -1, SCORE_DIFF], marker.astype(np.float32))
    assert (seq[:, -1, SCORE_DIFF] <= marker).all()


def test_windows_do_not_bleed_across_games() -> None:
    """The short game sits after a long one in row order; its history must be empty."""
    df = synthetic_frame()
    marker = np.arange(df.height, dtype=np.float64) + 1.0
    df = df.with_columns(pl.Series("ig__score_diff_home", marker))
    seq, mask, _ = build_sequences(df, k=DEFAULT_K, stats=RAW_STATS)

    short = np.flatnonzero(df.get_column("meta__game_id").to_numpy() == "G_short")
    first = short[0]
    assert mask[first].sum() == 1.0, "the first play of a game has one valid step"
    assert np.count_nonzero(seq[first, :-1, :]) == 0
    last = short[-1]
    assert mask[last].sum() == len(short)


def test_window_construction_is_independent_of_input_row_order() -> None:
    df = synthetic_frame()
    marker = np.arange(df.height, dtype=np.float64) + 1.0
    df = df.with_columns(pl.Series("ig__score_diff_home", marker))
    seq, mask, _ = build_sequences(df, k=8, stats=RAW_STATS)

    shuffled = df.sample(fraction=1.0, shuffle=True, seed=11)
    s_seq, s_mask, _ = build_sequences(shuffled, k=8, stats=RAW_STATS)

    order = shuffled.get_column("ig__score_diff_home").to_numpy().astype(int) - 1
    assert np.array_equal(s_seq, seq[order])
    assert np.array_equal(s_mask, mask[order])


def test_short_window_keeps_only_the_most_recent_plays() -> None:
    df = synthetic_frame()
    marker = np.arange(df.height, dtype=np.float64) + 1.0
    df = df.with_columns(pl.Series("ig__score_diff_home", marker))
    seq, mask, _ = build_sequences(df, k=3, stats=RAW_STATS)
    row = 10  # eleventh play of the long game
    assert mask[row].tolist() == [1.0, 1.0, 1.0]
    assert seq[row, :, SCORE_DIFF].tolist() == pytest.approx([9.0, 10.0, 11.0])


def test_build_sequences_rejects_a_nonsense_window() -> None:
    with pytest.raises(ValueError, match="k must be"):
        build_sequences(synthetic_frame(), k=0)


def test_model_order_right_pads_the_mask_for_cudnn() -> None:
    """cuDNN's fused GRU rejects anything but a right-padded mask, and Keras
    raises rather than falling back — so this fails on a GPU and passes on CPU
    if it ever regresses. Assert the shape of the mask itself instead."""
    df = synthetic_frame()
    seq, mask, _ = build_sequences(df, k=DEFAULT_K, stats=RAW_STATS)
    assert (mask[:, -1] == 1.0).all(), "chronological windows are padded on the left"

    m_seq, m_mask = _model_order(seq, mask)
    for row in m_mask:
        valid = int(row.sum())
        assert (row[:valid] == 1.0).all()
        assert (row[valid:] == 0.0).all()
    # Newest play first, and the values themselves are untouched.
    assert np.array_equal(m_seq[:, 0, :], seq[:, -1, :])
    assert np.array_equal(m_seq[:, ::-1, :], seq)


def test_seq_model_inputs_are_right_padded(frame: pl.DataFrame) -> None:
    model = tiny_seq()
    meta = frame.select("meta__game_id", "meta__play_id")
    inputs = model._inputs(features_of(frame), meta, fit=True)
    for row in inputs["play_mask"]:
        valid = int(row.sum())
        assert valid >= 1
        assert (row[:valid] == 1.0).all() and (row[valid:] == 0.0).all()


def test_pregame_columns_exclude_the_live_state(frame: pl.DataFrame) -> None:
    cols = pregame_columns(frame)
    assert cols and not any(c.startswith(("ig__", "bi__", "meta__", "label__")) for c in cols)
    assert "ctx__temp_f" in cols


# --------------------------------------------------------------------------- #
# MLP: fit / predict / save / load
# --------------------------------------------------------------------------- #


def tiny_mlp(**kwargs: object) -> MLPWinProb:
    params: dict[str, object] = {
        "hidden": (8, 4),
        "epochs": 2,
        "patience": 1,
        "batch_size": 64,
        # Off by default here so the MC-dropout tests isolate dropout: input
        # noise is the model's *other* stochastic component at training time.
        "input_noise": 0.0,
        "prefer_gpu": False,
        "mixed_precision": False,
    }
    params.update(kwargs)
    return MLPWinProb(**params)  # type: ignore[arg-type]


def test_mlp_satisfies_the_model_contract() -> None:
    assert isinstance(tiny_mlp(), WinProbModel)


def test_mlp_fit_predict_save_load_round_trip(tmp_path: Path, frame: pl.DataFrame) -> None:
    X, y = features_of(frame), labels_of(frame)
    model = tiny_mlp()
    result = model.fit(X, y, X, y)

    assert isinstance(result, FitResult)
    assert result.model_name == "tf_mlp"
    assert result.n_train == frame.height and result.n_val == frame.height
    assert set(result.metrics) >= {"log_loss", "brier", "accuracy", "ece"}
    assert result.feature_names == model.preprocessor.feature_names

    p = model.predict_proba(X)
    assert p.shape == (frame.height,)
    assert ((p > 0.0) & (p < 1.0)).all()

    model.save(tmp_path / "artifact")
    assert (tmp_path / "artifact" / "model.keras").is_file()
    back = MLPWinProb.load(tmp_path / "artifact")
    assert back.version == model.version
    assert np.allclose(back.predict_proba(X), p, atol=1e-6)


def test_mlp_writes_a_manifest_with_the_conventional_fields(
    tmp_path: Path, frame: pl.DataFrame
) -> None:
    X, y = features_of(frame), labels_of(frame)
    model = tiny_mlp()
    result = model.fit(X, y, X, y)
    path = write_manifest(
        tmp_path,
        name=model.name,
        version=result.version,
        train_seasons=[2020, 2021],
        feature_names=result.feature_names,
        metrics=result.metrics,
    )
    payload = json.loads(path.read_text())
    assert set(payload) >= {
        "name",
        "version",
        "git_sha",
        "train_seasons",
        "features",
        "metrics",
        "created_at",
    }
    assert payload["name"] == "tf_mlp"
    assert payload["version"] == result.version


def test_mc_dropout_disagrees_with_itself_when_dropout_is_on(frame: pl.DataFrame) -> None:
    X, y = features_of(frame), labels_of(frame)
    model = tiny_mlp(dropout=0.5)
    model.fit(X, y)

    mean, std = model.predict_with_uncertainty(X, n_samples=25)
    assert mean.shape == std.shape == (frame.height,)
    assert (std >= 0.0).all()
    assert (std > 0.0).mean() > 0.9, "dropout on: the passes must not all agree"
    assert ((mean > 0.0) & (mean < 1.0)).all()

    # n_samples > 1 on predict_proba is the same MC average, so it lands near
    # the mean rather than on the deterministic pass.
    mc = model.predict_proba(X, n_samples=25)
    assert np.abs(mc - mean).mean() < 0.05


def test_mc_dropout_is_deterministic_when_the_stochastic_parts_are_off(
    frame: pl.DataFrame,
) -> None:
    X, y = features_of(frame), labels_of(frame)
    model = tiny_mlp(dropout=0.0, input_noise=0.0)
    model.fit(X, y)
    _, std = model.predict_with_uncertainty(X, n_samples=5)
    assert np.allclose(std, 0.0)


def test_input_noise_alone_also_spreads_the_mc_samples(frame: pl.DataFrame) -> None:
    """Input jitter is a stochastic layer too, so it feeds the uncertainty."""
    X, y = features_of(frame), labels_of(frame)
    model = tiny_mlp(dropout=0.0, input_noise=0.5)
    model.fit(X, y)
    _, std = model.predict_with_uncertainty(X, n_samples=20)
    assert (std > 0.0).mean() > 0.9


def test_mlp_refuses_leaky_inputs(frame: pl.DataFrame) -> None:
    model = tiny_mlp()
    with pytest.raises(AssertionError, match="label__/meta__"):
        model.fit(frame, labels_of(frame))
    leaky = features_of(frame).with_columns(pl.lit(1.0).alias("spread_line"))
    with pytest.raises(AssertionError):
        model.fit(leaky, labels_of(frame))


def test_mlp_predict_before_fit_is_an_error(frame: pl.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        tiny_mlp().predict_proba(features_of(frame))


# --------------------------------------------------------------------------- #
# Sequence model: fit / predict / save / load
# --------------------------------------------------------------------------- #


def tiny_seq(**kwargs: object) -> SeqWinProb:
    params: dict[str, object] = {
        "k": 8,
        "play_embed": 4,
        "gru_units": 8,
        "pregame_units": 8,
        "head_units": 8,
        "epochs": 2,
        "patience": 1,
        "batch_size": 64,
        "prefer_gpu": False,
        "mixed_precision": False,
    }
    params.update(kwargs)
    return SeqWinProb(**params)  # type: ignore[arg-type]


def test_seq_satisfies_the_model_contract() -> None:
    assert isinstance(tiny_seq(), WinProbModel)


def test_seq_fit_predict_save_load_round_trip(tmp_path: Path, frame: pl.DataFrame) -> None:
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select("meta__game_id", "meta__play_id")
    model = tiny_seq()
    result = model.fit(X, y, X, y, meta=meta, meta_val=meta)

    assert result.model_name == "tf_seq"
    assert result.extra["k"] == 8
    assert result.feature_names[: len(PLAY_CHANNELS)] == list(PLAY_CHANNELS)
    assert result.metrics["log_loss"] > 0.0

    p = model.predict_proba(X, meta)
    assert p.shape == (frame.height,)
    assert ((p > 0.0) & (p < 1.0)).all()

    model.save(tmp_path / "artifact")
    back = SeqWinProb.load(tmp_path / "artifact")
    assert back.channel_stats == model.channel_stats
    assert back.pregame_names == model.pregame_names
    assert np.allclose(back.predict_proba(X, meta), p, atol=1e-6)


def test_seq_takes_the_game_key_separately_from_the_features(frame: pl.DataFrame) -> None:
    """Windowing needs a game id, but ``meta__`` may never ride along inside X."""
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select("meta__game_id", "meta__play_id")
    model = tiny_seq()
    model.fit(X, y, meta=meta)
    with_key = pl.concat([X, meta.select("meta__game_id")], how="horizontal")
    with pytest.raises(AssertionError, match="label__/meta__"):
        model.predict_proba(with_key)
    assert model.predict_proba(X, meta).shape == (frame.height,)


def test_seq_masking_makes_the_padded_steps_inert(frame: pl.DataFrame) -> None:
    """Perturbing masked slots must not move a prediction."""
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select("meta__game_id", "meta__play_id")
    model = tiny_seq()
    model.fit(X, y, meta=meta)

    inputs = model._inputs(X, meta, fit=False)
    baseline = model._predict(inputs)

    noisy = {k: v.copy() for k, v in inputs.items()}
    rng = np.random.default_rng(5)
    padding = noisy["play_mask"] == 0.0
    noise = rng.normal(0, 5.0, noisy["plays"].shape).astype(np.float32)
    noisy["plays"] = np.where(padding[:, :, None], noise, noisy["plays"])
    assert padding.any(), "the fixture must contain padded windows"
    assert np.allclose(model._predict(noisy), baseline, atol=1e-5)


def test_seq_refuses_leaky_inputs(frame: pl.DataFrame) -> None:
    model = tiny_seq()
    with pytest.raises(AssertionError, match="label__/meta__"):
        model.fit(frame, labels_of(frame))


def test_new_version_is_a_utc_timestamp() -> None:
    version = new_version()
    assert len(version) == 15 and version[8] == "T"
    assert version.replace("T", "").isdigit()


# --------------------------------------------------------------------------- #
# ADR 0003: game-aware training for the Keras models
# --------------------------------------------------------------------------- #


def q4_frame(seed: int = 5) -> pl.DataFrame:
    """A synthetic frame whose clock actually runs through the fourth quarter."""
    df = synthetic_frame(seed, lengths={"G_a": 30, "G_b": 24, "G_c": 18})
    # Compress every game's clock into 880..20 seconds of Q4 so all three marks
    # are reachable; ``synthetic_frame`` alone never leaves the first half.
    return df.with_columns(
        pl.lit(4, dtype=pl.Int64).alias("meta__period"),
        (880 - pl.int_range(pl.len()).over("meta__game_id") * 30)
        .clip(20, 880)
        .cast(pl.Float64)
        .alias("meta__game_seconds_remaining"),
    )


def test_seq_v1_feature_set_narrows_the_tabular_branch() -> None:
    from nfl.models.feature_sets import feature_set_columns

    frame = q4_frame()
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select("meta__game_id", "meta__play_id")

    wide = tiny_seq()
    wide.fit(X, y, meta=meta)
    narrow = tiny_seq(feature_set="v1")
    narrow.fit(X, y, meta=meta)

    assert narrow.pregame_names == feature_set_columns(X, "v1")
    assert wide.pregame_names != narrow.pregame_names
    # ``pregame_columns`` deliberately excludes the live state; the v1 set
    # deliberately includes it.
    assert any(c.startswith("ig__") for c in narrow.pregame_names)
    assert not any(c.startswith("ig__") for c in wide.pregame_names)


def test_seq_q4_selection_metric_logs_and_stops_on_the_q4_rows() -> None:
    from nfl.models.game_aware import Q4_LOG_KEY, game_weights, q4_selection

    frame = q4_frame()
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select(
        "meta__game_id", "meta__play_id", "meta__period", "meta__game_seconds_remaining"
    )
    selection = q4_selection(meta)
    assert len(selection) == 3 * 3  # three games, three marks

    model = tiny_seq(feature_set="v1", selection_metric="q4", epochs=3)
    result = model.fit(
        X,
        y,
        X,
        y,
        meta=meta,
        meta_val=meta,
        sample_weight=game_weights(meta),
        val_selection=selection,
    )
    assert Q4_LOG_KEY in model.history
    assert len(model.history[Q4_LOG_KEY]) >= 1
    assert result.extra["selection_metric"] == "q4"
    assert result.extra["sample_weighted"] is True
    assert result.extra["feature_set"] == "v1"


def test_seq_q4_without_a_selection_raises() -> None:
    frame = q4_frame()
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select("meta__game_id", "meta__play_id")
    model = tiny_seq(selection_metric="q4", epochs=1)
    with pytest.raises(ValueError, match="needs a validation set"):
        model.fit(X, y, X, y, meta=meta, meta_val=meta)


def test_mlp_q4_selection_metric_logs_the_key() -> None:
    from nfl.models.game_aware import Q4_LOG_KEY, game_weights, q4_selection

    frame = q4_frame(seed=7)
    X, y = features_of(frame), labels_of(frame)
    meta = frame.select(
        "meta__game_id", "meta__play_id", "meta__period", "meta__game_seconds_remaining"
    )
    model = MLPWinProb(
        hidden=(8,),
        epochs=3,
        batch_size=32,
        patience=2,
        prefer_gpu=False,
        mixed_precision=False,
        selection_metric="q4",
    )
    result = model.fit(
        X, y, X, y, sample_weight=game_weights(meta), val_selection=q4_selection(meta)
    )
    assert Q4_LOG_KEY in model.history
    assert result.extra["selection_metric"] == "q4"
    assert result.extra["sample_weighted"] is True


def test_mlp_rejects_an_unknown_selection_metric() -> None:
    with pytest.raises(ValueError, match="selection_metric"):
        MLPWinProb(selection_metric="brier")
