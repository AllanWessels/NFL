"""Season splits, Q4 slices, and the market-collapse placeholder."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.features import splits as splits_mod
from nfl.features.assemble import assemble_dataset
from nfl.features.splits import (
    DEFAULT_SPLIT,
    SeasonSplit,
    add_split_column,
    not_collapsed,
    q4_at,
    q4_window,
    split_frames,
    train_frame,
    val_frame,
)

# ``splits.test_frame`` must not be imported under a ``test_`` name here: pytest
# would collect it as a test case. ``holdout_frame`` is the alias for exactly
# this situation.
holdout_frame = splits_mod.holdout_frame


@pytest.fixture
def seasons_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta__season": [2019, 2022, 2023, 2024, 2025],
            "meta__game_id": [f"g{i}" for i in range(5)],
        }
    )


def test_default_split_boundaries(seasons_frame: pl.DataFrame) -> None:
    train, val, test = split_frames(seasons_frame)
    assert train.get_column("meta__season").to_list() == [2019, 2022]
    assert val.get_column("meta__season").to_list() == [2023]
    assert test.get_column("meta__season").to_list() == [2024, 2025]
    assert train.height + val.height + test.height == seasons_frame.height


def test_splits_are_configurable(seasons_frame: pl.DataFrame) -> None:
    split = SeasonSplit(train_through=2019, val_season=2022, test_from=2023)
    assert train_frame(seasons_frame, split).height == 1
    assert val_frame(seasons_frame, split).height == 1
    assert holdout_frame(seasons_frame, split).height == 3


def test_a_season_in_no_split_is_dropped() -> None:
    frame = pl.DataFrame({"meta__season": [2021]})
    split = SeasonSplit(train_through=2019, val_season=2020, test_from=2022)
    train, val, test = split_frames(frame, split)
    assert train.height == val.height == test.height == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_through": 2023, "val_season": 2023, "test_from": 2024},
        {"train_through": 2022, "val_season": 2024, "test_from": 2023},
    ],
)
def test_out_of_order_split_definitions_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SeasonSplit(**kwargs)


def test_split_column_is_meta_prefixed(seasons_frame: pl.DataFrame) -> None:
    tagged = add_split_column(seasons_frame)
    assert tagged.get_column("meta__split").to_list() == ["train", "train", "val", "test", "test"]


def test_split_works_on_a_raw_frame_without_the_meta_prefix() -> None:
    frame = pl.DataFrame({"season": [2021, 2024]})
    assert train_frame(frame).height == 1
    assert holdout_frame(frame).height == 1


def test_missing_season_column_raises() -> None:
    with pytest.raises(KeyError, match="meta__season"):
        train_frame(pl.DataFrame({"x": [1]}), DEFAULT_SPLIT)


# --------------------------------------------------------------------------- #
# Q4 slices
# --------------------------------------------------------------------------- #


@pytest.fixture
def clock_frame() -> pl.DataFrame:
    """Two games, one clock column, Q3 -> Q4 -> OT."""
    return pl.DataFrame(
        {
            "meta__game_id": ["a"] * 5 + ["b"] * 3,
            "meta__play_id": [1, 2, 3, 4, 5, 1, 2, 3],
            "meta__period": [3, 4, 4, 4, 5, 3, 4, 4],
            "meta__game_seconds_remaining": [1000, 900, 310, 120, 0, 1200, 400, 200],
        }
    )


def test_q4_at_takes_the_last_row_before_the_mark(clock_frame: pl.DataFrame) -> None:
    out = q4_at(clock_frame, 300)
    assert out.height == 2
    assert out.get_column("meta__game_id").to_list() == ["a", "b"]
    # 310 is the last snapshot at or before 5:00 left; 120 is after it.
    assert out.get_column("meta__game_seconds_remaining").to_list() == [310, 400]


def test_q4_at_excludes_earlier_quarters_and_overtime(clock_frame: pl.DataFrame) -> None:
    out = q4_at(clock_frame, 0)
    assert out.get_column("meta__period").unique().to_list() == [4]
    assert q4_at(clock_frame, 0, include_overtime=True).get_column(
        "meta__game_seconds_remaining"
    ).to_list() == [0, 200]


def test_q4_at_tolerance_drops_games_that_are_too_far_from_the_mark(
    clock_frame: pl.DataFrame,
) -> None:
    out = q4_at(clock_frame, 300, tolerance=20)
    assert out.get_column("meta__game_id").to_list() == ["a"]  # b's nearest row is 400


def test_q4_at_skips_games_that_never_reach_the_mark() -> None:
    frame = pl.DataFrame(
        {
            "meta__game_id": ["a"],
            "meta__play_id": [1],
            "meta__period": [4],
            "meta__game_seconds_remaining": [100],
        }
    )
    assert q4_at(frame, 300).height == 0


def test_q4_at_rejects_a_mark_outside_the_quarter(clock_frame: pl.DataFrame) -> None:
    with pytest.raises(ValueError):
        q4_at(clock_frame, 1200)


def test_q4_window_is_inclusive_and_order_agnostic(clock_frame: pl.DataFrame) -> None:
    out = q4_window(clock_frame, 0, 310)
    assert sorted(out.get_column("meta__game_seconds_remaining").to_list()) == [120, 200, 310]
    assert q4_window(clock_frame, 310, 0).height == out.height


def test_q4_slices_work_on_the_assembled_dataset(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    df = assemble_dataset([2024], pbp=pbp, schedules=schedules)
    at_five = q4_at(df, 300)
    assert at_five.height == 1
    assert at_five.get_column("meta__game_seconds_remaining")[0] >= 300
    window = q4_window(df, 0, 300)
    assert 0 < window.height < df.height
    assert window.get_column("meta__game_seconds_remaining").max() <= 300


# --------------------------------------------------------------------------- #
# not_collapsed
# --------------------------------------------------------------------------- #


def test_not_collapsed_is_a_no_op_without_the_market_column() -> None:
    frame = pl.DataFrame({"meta__game_id": ["a"], "bi__x": [1.0]})
    assert not_collapsed(frame).equals(frame)


def test_not_collapsed_can_insist_on_the_column() -> None:
    frame = pl.DataFrame({"meta__game_id": ["a"]})
    with pytest.raises(KeyError, match="market"):
        not_collapsed(frame, missing="raise")


def test_not_collapsed_filters_resolved_games_and_keeps_unknowns() -> None:
    frame = pl.DataFrame(
        {
            "meta__game_id": ["a", "b", "c", "d"],
            "market_prob_home": [0.5, 0.93, 0.02, None],
        }
    )
    kept = not_collapsed(frame).get_column("meta__game_id").to_list()
    assert kept == ["a", "d"]


def test_not_collapsed_bounds_are_validated() -> None:
    with pytest.raises(ValueError):
        not_collapsed(pl.DataFrame({"market_prob_home": [0.5]}), bounds=(0.9, 0.1))
