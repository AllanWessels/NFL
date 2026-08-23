"""Tests for the ADR 0003 feature-set selectors.

Two levels: hand-made frames pin the *rules* (what each name means, what order
columns come back in, how a missing column is handled), and the real assembled
fixture pins the *counts* — that ``v1`` really is "in-play plus a compressed
prior" on the genuine 775-column table, not a name that happens to work on a
toy frame.
"""

from __future__ import annotations

import polars as pl
import pytest

from nfl.models.artifacts import assert_model_input_frame
from nfl.models.feature_sets import (
    FEATURE_SET_NAMES,
    PRIOR_COLUMNS,
    describe,
    feature_set_columns,
    inplay_columns,
    prior_columns,
    select_features,
)
from tests.models_fixture import assembled_games


@pytest.fixture(scope="module")
def games() -> pl.DataFrame:
    return assembled_games()


def toy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta__game_id": ["g1"],
            "meta__season": [2024],
            "label__home_win": [1.0],
            "bi__score_diff_home": [3.0],
            "ig__epa_total_home": [1.5],
            "ctx__is_indoors": [1],
            "home_tm__elo_win_prob": [0.6],
            "home_tm__sos_elo_played_l4": [1500.0],
            "home_pl__wr1_position_rank": [2.0],
        }
    )


# --------------------------------------------------------------------------- #
# Selection rules
# --------------------------------------------------------------------------- #


def test_inplay_is_bi_and_ig_only():
    assert inplay_columns(toy()) == ["bi__score_diff_home", "ig__epa_total_home"]


def test_prior_follows_the_declared_order_not_the_frame_order():
    df = toy().select(["home_tm__elo_win_prob", "ctx__is_indoors"])
    # ctx__is_indoors comes second in the frame but seventh in PRIOR_COLUMNS,
    # and the declared order is what wins.
    assert prior_columns(df) == ["home_tm__elo_win_prob", "ctx__is_indoors"]
    assert prior_columns(df) == [c for c in PRIOR_COLUMNS if c in df.columns]


def test_v1_is_inplay_plus_prior():
    df = toy()
    assert feature_set_columns(df, "v1") == inplay_columns(df) + prior_columns(df)


def test_full_is_every_feature_column_and_no_meta_or_label():
    df = toy()
    cols = feature_set_columns(df, "full")
    assert "home_tm__sos_elo_played_l4" in cols
    assert not [c for c in cols if c.startswith(("meta__", "label__"))]


def test_v1_excludes_the_bulky_game_constant_families():
    cols = feature_set_columns(toy(), "v1")
    assert "home_tm__sos_elo_played_l4" not in cols
    assert "home_pl__wr1_position_rank" not in cols


def test_missing_prior_columns_are_skipped_not_raised():
    df = toy().drop("home_tm__elo_win_prob")
    assert "home_tm__elo_win_prob" not in prior_columns(df)
    assert "ctx__is_indoors" in feature_set_columns(df, "v1")


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown feature set"):
        feature_set_columns(toy(), "everything")


def test_name_is_case_and_whitespace_insensitive():
    assert feature_set_columns(toy(), " V1 ") == feature_set_columns(toy(), "v1")


def test_a_set_that_matches_nothing_raises():
    df = pl.DataFrame({"meta__game_id": ["g"], "label__home_win": [1.0]})
    with pytest.raises(ValueError, match="matched no columns"):
        feature_set_columns(df, "inplay")


def test_select_features_returns_a_legal_design_matrix():
    out = select_features(toy(), "v1")
    assert out.columns == feature_set_columns(toy(), "v1")
    assert_model_input_frame(out, context="feature_sets test")


def test_select_features_defaults_to_v1():
    assert select_features(toy()).columns == select_features(toy(), "v1").columns


def test_describe_lists_every_name():
    table = describe(toy())
    assert table.get_column("feature_set").to_list() == list(FEATURE_SET_NAMES)
    counts = dict(zip(*table.to_dict(as_series=False).values(), strict=True))
    assert counts["v1"] == counts["inplay"] + counts["prior"]
    assert counts["full"] > counts["v1"]


# --------------------------------------------------------------------------- #
# Against the real assembled dataset
# --------------------------------------------------------------------------- #


def test_real_dataset_sizes_match_the_adr(games: pl.DataFrame):
    n_inplay = len(inplay_columns(games))
    n_prior = len(prior_columns(games))
    n_full = len(feature_set_columns(games, "full"))
    # ADR 0003: 245 in-play columns, at most 10 prior, out of ~775 total.
    assert n_inplay == 245
    assert n_prior <= 10
    assert n_full > 700
    assert len(feature_set_columns(games, "v1")) == n_inplay + n_prior


def test_every_prior_column_exists_in_the_real_schema(games: pl.DataFrame):
    missing = [c for c in PRIOR_COLUMNS if c not in games.columns]
    assert missing == [], f"prior columns absent from the dataset: {missing}"


def test_v1_selection_on_the_real_frame_is_model_input_safe(games: pl.DataFrame):
    assert_model_input_frame(select_features(games, "v1"), context="v1 on real frame")
