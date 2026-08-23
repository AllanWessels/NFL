"""``XGBoostWinProb``: encoding, nulls, the round trip, importance, ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak
from nfl.features.assemble import LABEL_COLUMN
from nfl.models.artifacts import read_manifest
from nfl.models.base import FitResult, WinProbModel
from nfl.models.data import design_matrix, labels, load_model_frames
from nfl.models.feature_sets import feature_set_columns
from nfl.models.gbdt import (
    MAX_CATEGORY_CARDINALITY,
    XGBoostWinProb,
    feature_group_ablation,
    group_prefixes,
    train_xgboost,
)
from tests.models_fixture import assembled_games, write_tiny_dataset

# Every test trains on the CPU: CI has no GPU, and a 356-row fit gains nothing
# from one. ``test_device_auto_resolves`` covers the auto/fallback path.
#
# ``nthread`` is pinned low on purpose. XGBoost otherwise grabs every core it can
# find, and two dozen threads fighting over a 356-row histogram is slower than
# two *and* saturates the machine the suite is sharing.
THREADS = {"nthread": 2}
CPU = {
    "device": "cpu",
    "num_boost_round": 20,
    "early_stopping_rounds": 5,
    "params": THREADS,
}


@pytest.fixture(scope="module")
def games() -> pl.DataFrame:
    return assembled_games()


@pytest.fixture(scope="module")
def xy(games: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    return design_matrix(games), labels(games)


@pytest.fixture
def fitted(xy: tuple[pl.DataFrame, np.ndarray]) -> XGBoostWinProb:
    X, y = xy
    model = XGBoostWinProb(**CPU)
    model.fit(X, y, X, y)
    return model


# --------------------------------------------------------------------------- #
# Fit / predict
# --------------------------------------------------------------------------- #


def test_fit_returns_a_populated_fit_result(xy: tuple[pl.DataFrame, np.ndarray]) -> None:
    X, y = xy
    model = XGBoostWinProb(**CPU)
    result = model.fit(X, y, X, y)
    assert isinstance(result, FitResult)
    assert result.version and result.model_name == model.name
    assert result.n_train == X.height
    assert result.extra["device"] == "cpu"
    assert result.extra["n_trees"] >= 1
    assert "val_log_loss" in result.metrics
    assert result.feature_names == list(model.encoding.columns)


def test_model_satisfies_the_winprobmodel_protocol() -> None:
    assert isinstance(XGBoostWinProb(), WinProbModel)


def test_predictions_are_probabilities(
    fitted: XGBoostWinProb, xy: tuple[pl.DataFrame, np.ndarray]
) -> None:
    probs = fitted.predict_proba(xy[0])
    assert probs.shape == (xy[0].height,)
    assert np.all((probs > 0.0) & (probs < 1.0))


def test_predicting_before_fitting_is_an_error(xy: tuple[pl.DataFrame, np.ndarray]) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        XGBoostWinProb().predict_proba(xy[0])


def test_soft_labels_need_no_special_handling(xy: tuple[pl.DataFrame, np.ndarray]) -> None:
    """``binary:logistic`` optimises against a target in [0, 1], ties included."""
    X, y = xy
    tied = y.copy()
    tied[: len(tied) // 4] = 0.5
    result = XGBoostWinProb(**CPU).fit(X, tied, X, tied)
    assert np.isfinite(result.metrics["train_log_loss"])


def test_device_auto_resolves_to_something_that_trains(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    model = XGBoostWinProb(
        device="auto", num_boost_round=5, early_stopping_rounds=3, params=THREADS
    )
    result = model.fit(X, y, X, y)
    assert result.extra["device"] in {"cpu", "cuda"}
    assert model.device_used == result.extra["device"]


# --------------------------------------------------------------------------- #
# Encoding: nulls and strings
# --------------------------------------------------------------------------- #


def test_nulls_are_passed_through_as_missing_not_imputed(
    fitted: XGBoostWinProb, xy: tuple[pl.DataFrame, np.ndarray]
) -> None:
    """The fixture genuinely has nulls, and an all-null row must still score."""
    X = xy[0]
    assert X.get_column("bi__down").null_count() > 0
    blank = X.head(1).with_columns(
        [pl.lit(None, dtype=X.schema[c]).alias(c) for c in fitted.encoding.columns]
    )
    probs = fitted.predict_proba(blank)
    assert probs.shape == (1,) and np.isfinite(probs[0])


def test_high_cardinality_string_features_are_dropped_not_encoded(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    """A quarterback-id column has hundreds of levels on the real dataset, and
    encoding it invites memorisation; a two-level ``surface`` is a real feature.
    The cap has to tell them apart."""
    X, y = xy
    n = X.height
    poisoned = X.with_columns(
        pl.Series("home_pl__made_up_player_id", [f"p{i}" for i in range(n)], dtype=pl.String),
        pl.Series(
            "home_tm__made_up_surface",
            ["grass" if i % 2 else "turf" for i in range(n)],
            dtype=pl.String,
        ),
    )
    assert n > MAX_CATEGORY_CARDINALITY

    model = XGBoostWinProb(**CPU)
    result = model.fit(poisoned, y)
    assert "home_pl__made_up_player_id" not in model.encoding.columns
    assert "home_pl__made_up_player_id" in result.extra["dropped_string_features"]
    assert model.encoding.categories["home_tm__made_up_surface"] == ["grass", "turf"]


def test_the_default_cardinality_cap_keeps_the_fixtures_small_categories(
    fitted: XGBoostWinProb,
) -> None:
    assert all(
        len(levels) <= MAX_CATEGORY_CARDINALITY for levels in fitted.encoding.categories.values()
    )


def test_low_cardinality_strings_become_categorical_features(
    fitted: XGBoostWinProb,
) -> None:
    assert fitted.encoding.categories, "surface/roof should be encoded as categories"
    types = dict(zip(fitted.encoding.columns, fitted.encoding.feature_types, strict=True))
    for name in fitted.encoding.categories:
        assert types[name] == "c"


def test_an_unseen_category_scores_as_missing_rather_than_crashing(
    fitted: XGBoostWinProb, xy: tuple[pl.DataFrame, np.ndarray]
) -> None:
    name = next(iter(fitted.encoding.categories))
    row = xy[0].head(1).with_columns(pl.lit("a-surface-nobody-has-played-on").alias(name))
    assert np.isfinite(fitted.predict_proba(row)[0])


def test_a_frame_missing_an_encoded_column_is_refused(
    fitted: XGBoostWinProb, xy: tuple[pl.DataFrame, np.ndarray]
) -> None:
    with pytest.raises(KeyError, match="missing"):
        fitted.predict_proba(xy[0].drop(fitted.encoding.columns[0]))


# --------------------------------------------------------------------------- #
# Importance
# --------------------------------------------------------------------------- #


def test_feature_importance_is_ranked_by_gain(fitted: XGBoostWinProb) -> None:
    imp = fitted.feature_importance("gain")
    assert imp.height > 0
    values = imp.get_column("importance").to_list()
    assert values == sorted(values, reverse=True)
    assert imp.get_column("rank").to_list() == list(range(1, imp.height + 1))
    assert set(imp.get_column("feature")) <= set(fitted.encoding.columns)


def test_group_importance_shares_sum_to_one(fitted: XGBoostWinProb) -> None:
    groups = fitted.group_importance()
    assert set(groups.columns) == {"group", "n_features", "importance", "share"}
    assert float(groups.get_column("share").sum()) == pytest.approx(1.0, abs=1e-6)


def test_group_prefixes_splits_a_paired_group() -> None:
    assert group_prefixes("home_tm__/away_tm__") == ("home_tm__", "away_tm__")
    assert group_prefixes("bi__") == ("bi__",)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_load_round_trip_reproduces_predictions_exactly(
    fitted: XGBoostWinProb, xy: tuple[pl.DataFrame, np.ndarray], tmp_path: Path
) -> None:
    before = fitted.predict_proba(xy[0])
    fitted.save(tmp_path / "artifact")
    reloaded = XGBoostWinProb.load(tmp_path / "artifact")

    assert reloaded.version == fitted.version
    assert reloaded.encoding.columns == fitted.encoding.columns
    assert reloaded.encoding.categories == fitted.encoding.categories
    assert reloaded.best_iteration == fitted.best_iteration
    assert np.allclose(reloaded.predict_proba(xy[0]), before)


def test_saving_an_unfitted_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        XGBoostWinProb().save(tmp_path)


# --------------------------------------------------------------------------- #
# The ADR 0001 gate
# --------------------------------------------------------------------------- #


def test_a_market_column_in_the_design_matrix_is_refused(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    with pytest.raises(MarketColumnLeak):
        XGBoostWinProb(**CPU).fit(X.with_columns(pl.lit(0.5).alias("vegas_wp")), y)


def test_a_market_column_in_the_validation_frame_is_refused(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    with pytest.raises(MarketColumnLeak):
        XGBoostWinProb(**CPU).fit(X, y, X.with_columns(pl.lit(-3.0).alias("spread_line")), y)


def test_the_label_column_is_refused_as_a_feature(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    with pytest.raises(AssertionError, match="label/meta"):
        XGBoostWinProb(**CPU).fit(X.with_columns(pl.Series(LABEL_COLUMN, y)), y)


# --------------------------------------------------------------------------- #
# Ablation and training entry point
# --------------------------------------------------------------------------- #


def test_ablation_reports_one_row_per_group_plus_the_full_reference(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    frames = load_model_frames(root=processed)

    table = feature_group_ablation(
        ["bi__", "ctx__"],
        frames=frames,
        device="cpu",
        num_boost_round=10,
        early_stopping_rounds=3,
        params=THREADS,
    )
    assert table.get_column("group").to_list() == ["full", "drop bi__", "drop ctx__"]
    full = table.filter(pl.col("group") == "full").row(0, named=True)
    assert full["delta_log_loss"] == pytest.approx(0.0)
    assert full["n_dropped"] == 0
    dropped = table.filter(pl.col("group") == "drop bi__").row(0, named=True)
    assert dropped["n_features"] < full["n_features"]
    assert dropped["n_dropped"] == full["n_features"] - dropped["n_features"]


def test_train_xgboost_writes_a_manifest_with_importance_and_slices(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    models_root = tmp_path / "models"

    model, result = train_xgboost(
        root=processed,
        models_root=models_root,
        device="cpu",
        num_boost_round=20,
        early_stopping_rounds=5,
        params=THREADS,
        top_k_importance=10,
    )

    artifact = models_root / model.name / result.version
    assert (artifact / "model.ubj").is_file()
    assert (artifact / "schema.json").is_file()
    manifest = read_manifest(artifact)
    assert manifest["kind"] == "xgboost"
    assert manifest["train_seasons"] == [2021, 2022]
    assert manifest["device"] in {"cpu", "cuda"}
    assert 0 < len(manifest["feature_importance_gain"]) <= 10
    assert {row["slice"] for row in manifest["val_slices"]} >= {
        "overall",
        "q4_600s",
        "q4_300s",
        "q4_120s",
        "not_collapsed",
    }
    assert manifest["feature_names"] == list(model.encoding.columns)

    reloaded = XGBoostWinProb.load(artifact)
    X, _ = load_model_frames(root=processed).as_xy("val")
    assert np.allclose(reloaded.predict_proba(X), model.predict_proba(X))


def test_stride_thins_the_training_frame_only(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    whole = load_model_frames(root=processed)
    thinned = load_model_frames(root=processed, stride=3)
    assert thinned.train.height == len(range(0, whole.train.height, 3))
    assert thinned.val.height == whole.val.height
    assert thinned.test.height == whole.test.height


# --------------------------------------------------------------------------- #
# ADR 0003: game-aware training
# --------------------------------------------------------------------------- #


def test_train_xgboost_v1_records_the_feature_set_and_q4_selection(tmp_path: Path) -> None:
    """The v1 configuration: narrowed features, game weights, Q4 early stopping."""
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    models_root = tmp_path / "models"

    model, result = train_xgboost(
        root=processed,
        models_root=models_root,
        device="cpu",
        num_boost_round=20,
        early_stopping_rounds=5,
        params=THREADS,
        top_k_importance=10,
        feature_set="v1",
        weight_by_game=True,
        selection_metric="q4",
    )

    frames = load_model_frames(root=processed)
    expected = feature_set_columns(frames.train, "v1")
    # The encoding may drop a high-cardinality string column, so the fitted
    # columns are a subset of the feature set, never a superset of it.
    assert set(model.encoding.columns) <= set(expected)
    assert len(model.encoding.columns) < len(feature_set_columns(frames.train, "full"))

    manifest = read_manifest(models_root / model.name / result.version)
    assert manifest["feature_set"] == "v1"
    assert manifest["selection_metric"] == "q4"
    assert manifest["weight_by_game"] is True
    assert manifest["val_q4_selection_log_loss"] > 0

    # The weighted train matrix carries one unit of evidence per game.
    n_games = frames.train.get_column("meta__game_id").n_unique()
    assert result.extra["effective_train_weight"] == pytest.approx(n_games)
    # Early stopping watched the Q4 eval set, so it is in the history.
    assert "val_q4" in result.extra["eval_history"]


def test_q4_selection_metric_without_a_selection_raises(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    frames = load_model_frames(root=processed)
    X, y = frames.as_xy("train")
    Xv, yv = frames.as_xy("val")
    model = XGBoostWinProb(
        device="cpu", num_boost_round=5, early_stopping_rounds=2, selection_metric="q4"
    )
    with pytest.raises(ValueError, match="needs a validation set"):
        model.fit(X, y, Xv, yv)


def test_sample_weight_length_is_checked(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    X, y = load_model_frames(root=processed).as_xy("train")
    model = XGBoostWinProb(device="cpu", num_boost_round=3)
    with pytest.raises(ValueError, match="sample_weight has"):
        model.fit(X, y, sample_weight=np.ones(3))
