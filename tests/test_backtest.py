"""Offline backtesting against a local ``WinProbModel`` double.

The double lives here on purpose: ``nfl/models/{naive,gbdt,tf_mlp,...}.py`` are
written by other agents, and this module must be able to prove the backtest
contract without any of them existing. It implements exactly the surface
:mod:`nfl.logging.backtest` uses, so if that surface widens these tests break
first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak
from nfl.features.assemble import feature_columns
from nfl.logging import evaluate as ev
from nfl.logging import ledger as ledger_mod
from nfl.logging.backtest import backtest, resolve_slices
from nfl.models.base import FitResult, WinProbModel


class ClockModel:
    """Predicts from the clock and score alone. Deliberately naive, fully known.

    ``P(home) = sigmoid(score_diff / 7)``, ignoring everything else, so a test
    can assert an exact probability without training anything.
    """

    name = "clock"

    def __init__(self) -> None:
        self.seen_columns: list[str] = []

    def fit(
        self,
        X: pl.DataFrame,  # noqa: N803 - the contract fixes this spelling
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,  # noqa: N803
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        return FitResult(self.name, "test", len(y), 0 if y_val is None else len(y_val))

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:  # noqa: N803
        self.seen_columns = list(X.columns)
        diff = X.get_column("score_diff").fill_null(0).cast(pl.Float64).to_numpy()
        return 1.0 / (1.0 + np.exp(-diff / 7.0))

    def save(self, path: Path) -> None:  # pragma: no cover - not exercised here
        path.write_text("clock")

    @classmethod
    def load(cls, path: Path) -> ClockModel:  # pragma: no cover - not exercised here
        return cls()


def test_the_double_satisfies_the_model_contract() -> None:
    assert isinstance(ClockModel(), WinProbModel)


@pytest.fixture
def dataset() -> pl.DataFrame:
    """Two games, three snapshots each, in the assembled-dataset shape."""
    games = ["2024_01_AA_BB", "2024_02_CC_DD"]
    clocks = [800, 400, 100]
    rows = []
    for g_i, game in enumerate(games):
        for c_i, clock in enumerate(clocks):
            rows.append(
                {
                    "meta__season": 2024,
                    "meta__week": g_i + 1,
                    "meta__game_id": game,
                    "meta__play_id": c_i + 1,
                    "meta__game_seconds_remaining": clock,
                    "meta__period": 4,
                    "meta__home_team": game.split("_")[3],
                    "meta__away_team": game.split("_")[2],
                    "meta__home_score": 21 + c_i,
                    "meta__away_score": 17,
                    "meta__posteam": "AA",
                    "meta__down": 2,
                    "meta__ydstogo": 7,
                    "score_diff": (4 + c_i) * (1 if g_i == 0 else -1),
                    "clock_frac": clock / 900.0,
                    "sometimes_null": None if c_i == 0 else 1.0,
                    "label__home_win": g_i == 0,
                }
            )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Schema equivalence — the whole point of the module
# --------------------------------------------------------------------------- #


def test_backtest_emits_exactly_the_ledger_schema(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset)
    assert list(out.columns) == list(ledger_mod.LEDGER_SCHEMA)
    assert out.schema == ledger_mod.empty_ledger().schema
    assert out.height == dataset.height


def test_a_backtest_frame_can_be_appended_to_the_real_ledger(
    dataset: pl.DataFrame, tmp_path: Path
) -> None:
    out = backtest(ClockModel(), dataset)
    ledger_mod.append(out, root=tmp_path)
    assert ledger_mod.read_ledger(tmp_path).height == dataset.height


def test_evaluate_scores_a_backtest_frame_the_same_way_it_scores_the_ledger(
    dataset: pl.DataFrame,
) -> None:
    out = backtest(ClockModel(), dataset)
    report = ev.build_report(out, source="backtest", with_market=False)
    assert report.models[0].model_name == "clock"
    assert report.models[0].slices[0].metrics.n == dataset.height


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #


def test_backtest_copies_the_snapshot_and_state_onto_each_row(dataset: pl.DataFrame) -> None:
    row = backtest(ClockModel(), dataset).sort("snapshot_play_index").row(0, named=True)
    assert row["game_id"] == "2024_01_AA_BB"
    assert row["season"] == 2024
    assert row["snapshot_game_seconds_remaining"] == 800
    assert row["snapshot_period"] == 4
    assert (row["home_score"], row["away_score"]) == (21, 17)
    assert (row["down"], row["ydstogo"]) == (2, 7)
    assert row["posteam"] == "AA"
    assert row["source"] == "backtest"
    assert row["is_live"] is False


def test_backtest_labels_and_scores_rows_from_the_dataset_label(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset)
    won = out.filter(pl.col("game_id") == "2024_01_AA_BB")
    lost = out.filter(pl.col("game_id") == "2024_02_CC_DD")

    assert won.get_column("resolved_home_win").to_list() == [True] * 3
    assert lost.get_column("resolved_home_win").to_list() == [False] * 3
    assert won.get_column("brier").null_count() == 0
    assert won.get_column("resolved_at_utc").null_count() == 0

    # p = sigmoid(4/7) for the first snapshot; brier must agree with it.
    p = float(won.sort("snapshot_play_index").get_column("p_home")[0])
    assert p == pytest.approx(1.0 / (1.0 + np.exp(-4 / 7)))
    assert float(won.sort("snapshot_play_index").get_column("brier")[0]) == pytest.approx(
        (p - 1) ** 2
    )


def test_backtest_counts_non_null_features_per_row(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset).sort("snapshot_play_index")
    n_features = len(feature_columns(dataset))

    assert out.get_column("features_total_count").to_list() == [n_features] * 6
    # The first snapshot of each game has a null feature; the rest are full.
    assert out.get_column("features_nonnull_count").to_list()[:2] == [n_features - 1] * 2
    assert out.get_column("features_nonnull_count").to_list()[-2:] == [n_features] * 2


def test_default_confidence_is_distance_from_the_coin_flip(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset)
    probs = out.get_column("p_home").to_numpy()
    assert out.get_column("confidence").to_numpy() == pytest.approx(np.abs(probs - 0.5) * 2)


def test_confidence_can_be_supplied_per_row(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset, confidence=np.full(6, 0.25))
    assert out.get_column("confidence").to_list() == [0.25] * 6


def test_market_columns_are_left_null_for_the_evaluator_to_fill(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset)
    for column in ledger_mod.EVAL_ONLY_COLUMNS:
        assert out.get_column(column).null_count() == out.height


# --------------------------------------------------------------------------- #
# Feature hygiene
# --------------------------------------------------------------------------- #


def test_no_meta_or_label_column_ever_reaches_the_model(dataset: pl.DataFrame) -> None:
    model = ClockModel()
    backtest(model, dataset)
    assert model.seen_columns == feature_columns(dataset)
    assert not [c for c in model.seen_columns if c.startswith(("meta__", "label__"))]


def test_a_market_column_in_the_dataset_stops_the_backtest(dataset: pl.DataFrame) -> None:
    poisoned = dataset.with_columns(pl.lit(-3.5).alias("spread_line"))
    with pytest.raises(MarketColumnLeak):
        backtest(ClockModel(), poisoned)


def test_a_dataset_with_no_features_is_a_clear_error() -> None:
    only_meta = pl.DataFrame({"meta__game_id": ["g"], "label__home_win": [True]})
    with pytest.raises(ValueError, match="no feature columns"):
        backtest(ClockModel(), only_meta)


def test_feature_hash_defaults_to_the_feature_set_anchor(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset)
    hashes = set(out.get_column("feature_hash").to_list())
    assert len(hashes) == 1
    assert hashes == {ledger_mod.feature_hash((), feature_columns(dataset))}


def test_hash_rows_gives_each_distinct_vector_its_own_hash(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset, hash_rows=True)
    hashes = out.get_column("feature_hash").to_list()
    assert len(set(hashes)) == dataset.height


def test_a_repeated_row_hashes_identically(dataset: pl.DataFrame) -> None:
    doubled = pl.concat([dataset.head(1), dataset.head(1)])
    hashes = backtest(ClockModel(), doubled, hash_rows=True).get_column("feature_hash").to_list()
    assert hashes[0] == hashes[1]


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #


def test_slices_restrict_which_rows_get_predicted(dataset: pl.DataFrame) -> None:
    assert backtest(ClockModel(), dataset, ["q4_120"]).height == 2  # one row per game
    assert backtest(ClockModel(), dataset, ["q4"]).height == 6


def test_slices_are_unioned_not_intersected(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset, ["q4_600", "q4_120"])
    # Two games x two marks, and the marks pick different rows.
    assert out.height == 4
    assert sorted(out.get_column("snapshot_game_seconds_remaining").unique().to_list()) == [
        400,
        800,
    ]


def test_a_row_selected_by_two_slices_is_predicted_once(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset, ["all", "q4", "q4_120"])
    assert out.height == dataset.height
    assert out.get_column("prediction_id").n_unique() == dataset.height


def test_an_unknown_slice_name_says_what_is_available() -> None:
    with pytest.raises(KeyError, match="unknown slice"):
        resolve_slices(["fourth_down_only"])


def test_slices_accept_a_custom_callable(dataset: pl.DataFrame) -> None:
    out = backtest(
        ClockModel(),
        dataset,
        {"first_game": lambda df: df.filter(pl.col("meta__game_id") == "2024_01_AA_BB")},
    )
    assert out.height == 3


def test_a_slice_that_matches_nothing_returns_an_empty_ledger_frame(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset, {"none": lambda df: df.head(0)})
    assert out.height == 0
    assert list(out.columns) == list(ledger_mod.LEDGER_SCHEMA)


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #


def test_batching_produces_the_same_probabilities_as_one_shot(dataset: pl.DataFrame) -> None:
    whole = backtest(ClockModel(), dataset).get_column("p_home").to_list()
    batched = backtest(ClockModel(), dataset, batch_size=2).get_column("p_home").to_list()
    assert batched == pytest.approx(whole)


def test_a_model_returning_the_wrong_number_of_rows_is_caught(dataset: pl.DataFrame) -> None:
    class Short(ClockModel):
        def predict_proba(self, X: pl.DataFrame) -> np.ndarray:  # noqa: N803
            return np.array([0.5])

    with pytest.raises(ValueError, match="probabilities for"):
        backtest(Short(), dataset)


def test_write_is_opt_in(dataset: pl.DataFrame, tmp_path: Path) -> None:
    backtest(ClockModel(), dataset, root=tmp_path)
    assert ledger_mod.read_ledger(tmp_path).height == 0

    backtest(ClockModel(), dataset, write=True, root=tmp_path)
    assert ledger_mod.read_ledger(tmp_path).height == dataset.height


def test_model_name_falls_back_to_the_model_attribute(dataset: pl.DataFrame) -> None:
    out = backtest(ClockModel(), dataset)
    assert out.get_column("model_name").unique().to_list() == ["clock"]
    named = backtest(ClockModel(), dataset, model_name="override", model_version="v9")
    assert named.get_column("model_name").unique().to_list() == ["override"]
    assert named.get_column("model_version").unique().to_list() == ["v9"]
