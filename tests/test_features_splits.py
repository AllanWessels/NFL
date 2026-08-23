"""Season splits, Q4 slices, and the market-collapse placeholder."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.features import splits as splits_mod
from nfl.features.assemble import assemble_dataset
from nfl.features.splits import (
    FALLBACK_SPLIT,
    SPLIT_ENV_VAR,
    SeasonSplit,
    add_split_column,
    default_split,
    latest_complete_season,
    not_collapsed,
    q4_at,
    q4_window,
    resolve_split,
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
        train_frame(pl.DataFrame({"x": [1]}), FALLBACK_SPLIT)


# --------------------------------------------------------------------------- #
# Rolling splits
# --------------------------------------------------------------------------- #


def _schedule(rows: list[tuple[int, str, int | None]]) -> pl.DataFrame:
    """``(season, game_type, result)`` rows as an nflverse-shaped schedule."""
    return pl.DataFrame(
        {
            "season": [r[0] for r in rows],
            "game_type": [r[1] for r in rows],
            "result": [r[2] for r in rows],
        },
        schema={"season": pl.Int32, "game_type": pl.String, "result": pl.Int32},
    )


def test_rolling_holds_out_the_two_newest_seasons() -> None:
    split = SeasonSplit.rolling(2025)
    assert (split.train_through, split.val_season, split.test_from) == (2023, 2024, 2025)


def test_rolling_split_frames_put_each_season_where_the_window_says() -> None:
    frame = pl.DataFrame({"meta__season": [2019, 2023, 2024, 2025]})
    train, val, test = split_frames(frame, SeasonSplit.rolling(2025))
    assert train.get_column("meta__season").to_list() == [2019, 2023]
    assert val.get_column("meta__season").to_list() == [2024]
    assert test.get_column("meta__season").to_list() == [2025]


def test_latest_complete_season_is_the_newest_finished_regular_season() -> None:
    schedules = _schedule(
        [(2023, "REG", 3), (2024, "REG", -7), (2024, "REG", 1), (2025, "REG", 10)]
    )
    assert latest_complete_season(schedules) == 2025


def test_a_season_still_being_played_is_not_complete() -> None:
    """The whole point: a half-finished season must never become val or test."""
    schedules = _schedule([(2024, "REG", 3), (2025, "REG", 7), (2025, "REG", None)])
    assert latest_complete_season(schedules) == 2024


def test_unplayed_playoffs_do_not_hold_a_season_back() -> None:
    """The regular season is what dates the split; January is not worth waiting for."""
    schedules = _schedule([(2025, "REG", 3), (2025, "REG", 7), (2025, "SB", None)])
    assert latest_complete_season(schedules) == 2025


def test_latest_complete_season_is_none_without_a_usable_schedule() -> None:
    assert latest_complete_season(pl.DataFrame({"nothing": [1]})) is None
    assert latest_complete_season(_schedule([(2025, "REG", None)])) is None


def test_default_split_reads_the_schedules_on_disk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SPLIT_ENV_VAR, raising=False)
    splits_mod.clear_split_cache()
    partition = tmp_path / "nflverse" / "schedules" / "season=2025"
    partition.mkdir(parents=True)
    _schedule([(2024, "REG", 1), (2025, "REG", 3)]).write_parquet(partition / "part-0.parquet")

    assert default_split(tmp_path) == SeasonSplit.rolling(2025)


def test_default_split_falls_back_when_there_are_no_schedules(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SPLIT_ENV_VAR, raising=False)
    splits_mod.clear_split_cache()
    assert default_split(tmp_path / "empty") == FALLBACK_SPLIT


def test_the_environment_variable_pins_the_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SPLIT_ENV_VAR, "2019/2020/2021")
    assert default_split() == SeasonSplit(2019, 2020, 2021)


def test_a_bad_pin_says_so_rather_than_silently_rolling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SPLIT_ENV_VAR, "last-tuesday")
    with pytest.raises(ValueError, match=SPLIT_ENV_VAR):
        default_split()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2019/2020/2021", SeasonSplit(2019, 2020, 2021)),
        ("2019,2020,2021", SeasonSplit(2019, 2020, 2021)),
        ("rolling:2025", SeasonSplit(2023, 2024, 2025)),
        ("2025", SeasonSplit(2023, 2024, 2025)),
    ],
)
def test_split_specs_parse(spec: str, expected: SeasonSplit) -> None:
    assert SeasonSplit.parse(spec) == expected


@pytest.mark.parametrize("spec", ["", "rolling", "2020/2021", "2020/2021/2022/2023"])
def test_bad_split_specs_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        SeasonSplit.parse(spec)


def test_resolve_split_prefers_what_it_is_given() -> None:
    explicit = SeasonSplit(2015, 2016, 2017)
    assert resolve_split(explicit) is explicit
    assert resolve_split("rolling:2025") == SeasonSplit.rolling(2025)
    assert resolve_split(None) == SeasonSplit(2022, 2023, 2024)  # the conftest pin


def test_a_split_round_trips_through_a_manifest_payload() -> None:
    split = SeasonSplit.rolling(2025)
    assert SeasonSplit.from_dict(split.to_dict()) == split
    assert SeasonSplit.from_dict({"train_through": 2023}) is None
    assert SeasonSplit.from_dict("nope") is None
    assert split.label == "<=2023/2024/2025+"


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
