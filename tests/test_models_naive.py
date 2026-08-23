"""``ScoreTimeLogistic``: the design matrix, the fit, the round trip, the gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak
from nfl.features.assemble import LABEL_COLUMN, feature_columns
from nfl.models.artifacts import read_manifest
from nfl.models.base import FitResult, WinProbModel
from nfl.models.data import design_matrix, labels, load_model_frames
from nfl.models.naive import (
    SOURCE_COLUMNS,
    TERM_NAMES,
    ScoreTimeLogistic,
    design_terms,
    train_naive,
)
from tests.models_fixture import assembled_games, write_tiny_dataset


@pytest.fixture(scope="module")
def games() -> pl.DataFrame:
    return assembled_games()


@pytest.fixture(scope="module")
def xy(games: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    return design_matrix(games), labels(games)


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #


def test_design_terms_emits_exactly_the_named_terms(xy: tuple[pl.DataFrame, np.ndarray]) -> None:
    terms = design_terms(xy[0])
    assert terms.columns == list(TERM_NAMES)
    assert terms.height == xy[0].height


def test_design_terms_preserves_nulls_for_later_imputation(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    """``bi__down`` is null on a kickoff; the raw term must still be null here."""
    X = xy[0]
    assert X.get_column("bi__down").null_count() > 0
    assert design_terms(X).get_column("down").null_count() > 0


def test_design_terms_names_the_columns_it_is_missing() -> None:
    with pytest.raises(KeyError, match="bi__score_diff_home"):
        design_terms(pl.DataFrame({"bi__down": [1.0]}))


def test_interaction_term_grows_as_the_clock_runs_out() -> None:
    """The same three-point lead is worth more with 10 seconds left than 3000."""
    X = pl.DataFrame(
        {c: [0.0, 0.0] for c in SOURCE_COLUMNS}
        | {
            "bi__score_diff_home": [3.0, 3.0],
            "bi__game_seconds_remaining": [3000.0, 10.0],
        }
    )
    term = design_terms(X).get_column("score_diff_per_sqrt_time").to_list()
    assert term[1] > term[0] * 10


# --------------------------------------------------------------------------- #
# Fit / predict
# --------------------------------------------------------------------------- #


def test_fit_returns_a_fit_result_with_metrics_and_features(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    model = ScoreTimeLogistic()
    result = model.fit(X, y, X, y)
    assert isinstance(result, FitResult)
    assert result.model_name == model.name
    assert result.version
    assert result.feature_names == list(SOURCE_COLUMNS)
    for key in ("train_log_loss", "val_log_loss", "val_ece", "val_auc"):
        assert key in result.metrics
    assert 0.0 < result.metrics["train_log_loss"] < 1.0


def test_model_satisfies_the_winprobmodel_protocol() -> None:
    assert isinstance(ScoreTimeLogistic(), WinProbModel)


def test_predictions_are_probabilities(xy: tuple[pl.DataFrame, np.ndarray]) -> None:
    X, y = xy
    model = ScoreTimeLogistic()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (X.height,)
    assert np.all(probs > 0.0) and np.all(probs < 1.0)


def test_predicting_before_fitting_is_an_error(xy: tuple[pl.DataFrame, np.ndarray]) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ScoreTimeLogistic().predict_proba(xy[0])


def test_a_tie_is_fitted_as_half_of_each_outcome(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    """A 0.5 label must neither crash sklearn nor be silently dropped."""
    X, y = xy
    tied = y.copy()
    tied[: len(tied) // 4] = 0.5
    model = ScoreTimeLogistic()
    result = model.fit(X, tied)
    assert np.isfinite(result.metrics["train_log_loss"])
    assert np.all(np.isfinite(model.predict_proba(X)))


def test_a_bigger_late_lead_raises_the_home_probability() -> None:
    """A sanity check on direction, fitted on a synthetic separable game."""
    rng = np.random.default_rng(11)
    n = 4000
    diff = rng.integers(-21, 22, n).astype(float)
    secs = rng.integers(0, 3600, n).astype(float)
    X = pl.DataFrame({c: np.zeros(n) for c in SOURCE_COLUMNS}).with_columns(
        pl.Series("bi__score_diff_home", diff),
        pl.Series("bi__game_seconds_remaining", secs),
    )
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * diff / np.sqrt(secs + 1)))).astype(float)
    model = ScoreTimeLogistic()
    model.fit(X, y)

    late = pl.DataFrame({c: [0.0, 0.0] for c in SOURCE_COLUMNS}).with_columns(
        pl.Series("bi__score_diff_home", [3.0, 14.0]),
        pl.Series("bi__game_seconds_remaining", [60.0, 60.0]),
    )
    probs = model.predict_proba(late)
    assert probs[1] > probs[0] > 0.5


def test_nulls_are_imputed_with_the_persisted_training_median(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    model = ScoreTimeLogistic()
    model.fit(X, y)
    all_null = pl.DataFrame(
        {c: pl.Series([None], dtype=pl.Float64) for c in SOURCE_COLUMNS},
    )
    probs = model.predict_proba(all_null)
    assert probs.shape == (1,)
    assert np.isfinite(probs[0])


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_save_load_round_trip_reproduces_predictions_exactly(
    xy: tuple[pl.DataFrame, np.ndarray], tmp_path: Path
) -> None:
    X, y = xy
    model = ScoreTimeLogistic()
    model.fit(X, y)
    before = model.predict_proba(X)

    model.save(tmp_path / "artifact")
    reloaded = ScoreTimeLogistic.load(tmp_path / "artifact")

    assert reloaded.version == model.version
    assert np.allclose(reloaded.predict_proba(X), before)
    assert reloaded.coefficients() == model.coefficients()


def test_saving_an_unfitted_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ScoreTimeLogistic().save(tmp_path)


# --------------------------------------------------------------------------- #
# The ADR 0001 gate
# --------------------------------------------------------------------------- #


def test_a_market_column_in_the_design_matrix_is_refused(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    poisoned = X.with_columns(pl.lit(-3.0).alias("spread_line"))
    with pytest.raises(MarketColumnLeak):
        ScoreTimeLogistic().fit(poisoned, y)


def test_a_third_party_win_probability_column_is_refused(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    poisoned = X.with_columns(pl.lit(0.7).alias("vegas_home_wp"))
    with pytest.raises(MarketColumnLeak):
        ScoreTimeLogistic().fit(poisoned, y)


def test_the_label_column_is_refused_as_a_feature(
    xy: tuple[pl.DataFrame, np.ndarray],
) -> None:
    X, y = xy
    with pytest.raises(AssertionError, match="label/meta"):
        ScoreTimeLogistic().fit(X.with_columns(pl.Series(LABEL_COLUMN, y)), y)


def test_meta_columns_are_refused_as_features(
    games: pl.DataFrame, xy: tuple[pl.DataFrame, np.ndarray]
) -> None:
    X, y = xy
    with pytest.raises(AssertionError, match="label/meta"):
        ScoreTimeLogistic().fit(X.with_columns(games.get_column("meta__game_id")), y)


def test_feature_columns_of_the_fixture_carry_no_market_column(games: pl.DataFrame) -> None:
    from nfl.data.market_blocklist import market_columns_in

    assert market_columns_in(feature_columns(games)) == []


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #


def test_train_naive_writes_a_manifest_and_a_loadable_model(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    models_root = tmp_path / "models"

    model, result = train_naive(root=processed, models_root=models_root)

    artifact = models_root / model.name / result.version
    assert (artifact / "model.joblib").is_file()
    manifest = read_manifest(artifact)
    assert manifest["name"] == model.name
    assert manifest["version"] == result.version
    assert manifest["train_seasons"] == [2021, 2022]
    assert manifest["feature_names"] == list(SOURCE_COLUMNS)
    assert manifest["kind"] == "logistic_regression"
    assert {row["slice"] for row in manifest["val_slices"]} >= {"overall", "not_collapsed"}

    reloaded = ScoreTimeLogistic.load(artifact)
    frames = load_model_frames(root=processed)
    X, _ = frames.as_xy("val")
    assert np.allclose(reloaded.predict_proba(X), model.predict_proba(X))


def test_train_naive_can_skip_saving(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_tiny_dataset(processed)
    _, result = train_naive(root=processed, models_root=tmp_path / "models", save=False)
    assert not (tmp_path / "models").exists()
    assert "val_slices" in result.extra
