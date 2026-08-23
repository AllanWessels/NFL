"""The shared leakage checks: delete-the-future, market gate, trivial-leak detector."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak
from nfl.features.assemble import LABEL_COLUMN, assemble_dataset
from nfl.features.leakage import (
    LeakageError,
    check_delete_the_future,
    check_no_market,
    check_no_trivial_leak,
    truncate_inputs,
)
from nfl.features.registry import make_builder


def _honest(pbp: pl.DataFrame) -> pl.DataFrame:
    """Cumulative score using plays strictly before t — the correct pattern."""
    return pbp.sort(["game_id", "play_id"]).select(
        "game_id",
        "play_id",
        pl.col("total_home_score")
        .shift(1)
        .over("game_id")
        .fill_null(0)
        .cast(pl.Float64)
        .alias("h__score"),
    )


def _peeks_at_the_next_play(pbp: pl.DataFrame) -> pl.DataFrame:
    """The classic bug: a feature that reads play t+1."""
    return pbp.sort(["game_id", "play_id"]).select(
        "game_id",
        "play_id",
        pl.col("total_home_score").shift(-1).over("game_id").cast(pl.Float64).alias("h__score"),
    )


def _uses_the_final_score(pbp: pl.DataFrame) -> pl.DataFrame:
    """A whole-game aggregate: every row learns how the game ended."""
    return pbp.sort(["game_id", "play_id"]).select(
        "game_id",
        "play_id",
        pl.col("total_home_score").max().over("game_id").cast(pl.Float64).alias("h__score"),
    )


# --------------------------------------------------------------------------- #
# truncate_inputs
# --------------------------------------------------------------------------- #


def test_truncation_is_inclusive_by_default(pbp: pl.DataFrame) -> None:
    cut = int(pbp.get_column("play_id").sort()[10])
    kept = truncate_inputs({"pbp": pbp}, cut)["pbp"]
    assert kept.get_column("play_id").max() == cut
    strict = truncate_inputs({"pbp": pbp}, cut, inclusive=False)["pbp"]
    assert strict.get_column("play_id").max() < cut


def test_frames_without_the_order_column_pass_through(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    out = truncate_inputs({"pbp": pbp, "schedules": schedules}, 100)
    assert out["schedules"].height == schedules.height
    assert out["pbp"].height < pbp.height


def test_only_restricts_which_frames_are_truncated(pbp: pl.DataFrame) -> None:
    out = truncate_inputs({"pbp": pbp}, 100, only=["other"])
    assert out["pbp"].height == pbp.height


# --------------------------------------------------------------------------- #
# check_delete_the_future
# --------------------------------------------------------------------------- #


def test_an_honest_builder_passes(pbp: pl.DataFrame) -> None:
    check = check_delete_the_future(_honest, {"pbp": pbp}, cut=2000)
    assert check.rows_compared > 0
    assert check.columns_compared == ("h__score",)
    assert check.keys == ("game_id", "play_id")


@pytest.mark.parametrize("builder", [_peeks_at_the_next_play, _uses_the_final_score])
def test_a_builder_that_reads_the_future_fails(builder, pbp: pl.DataFrame) -> None:
    with pytest.raises(LeakageError, match="changed earlier rows"):
        check_delete_the_future(builder, {"pbp": pbp}, cut=2000)


def test_the_error_names_the_offending_column(pbp: pl.DataFrame) -> None:
    with pytest.raises(LeakageError, match="h__score"):
        check_delete_the_future(_uses_the_final_score, {"pbp": pbp}, cut=2000)


def test_a_builder_spec_can_be_checked_directly(pbp: pl.DataFrame) -> None:
    spec = make_builder("honest", _honest, level="play", prefix="h__")
    assert check_delete_the_future(spec, {"pbp": pbp}, cut=2000).rows_compared > 0


def test_pre_game_builders_can_be_cut_on_kickoff() -> None:
    """The same check with a kickoff cut catches an end-of-season rank (taxonomy 2)."""
    from datetime import datetime

    games = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "team": ["KC", "KC", "KC"],
            "kickoff": [datetime(2024, 9, 8), datetime(2024, 9, 15), datetime(2024, 9, 22)],
            "points": [10.0, 20.0, 30.0],
        }
    )

    def as_of(games: pl.DataFrame) -> pl.DataFrame:
        return games.sort("kickoff").select(
            "game_id",
            "team",
            pl.col("points").cum_sum().shift(1).over("team").alias("tm__points_before"),
        )

    def whole_season(games: pl.DataFrame) -> pl.DataFrame:
        return games.select(
            "game_id", "team", pl.col("points").mean().over("team").alias("tm__points_before")
        )

    cut = datetime(2024, 9, 16)
    kwargs = {"order_col": "kickoff", "inclusive": False, "keys": ("game_id", "team")}
    assert check_delete_the_future(as_of, {"games": games}, cut, **kwargs).rows_compared == 2
    with pytest.raises(LeakageError):
        check_delete_the_future(whole_season, {"games": games}, cut, **kwargs)


def test_a_builder_that_invents_rows_when_truncated_fails(pbp: pl.DataFrame) -> None:
    def unstable(pbp: pl.DataFrame) -> pl.DataFrame:
        base = pbp.select("game_id", "play_id", pl.lit(1.0).alias("u__x"))
        if pbp.height < 50:  # pretend the builder pads short inputs
            base = pl.concat(
                [
                    base,
                    pl.DataFrame({"game_id": ["ghost"], "u__x": [1.0]}).with_columns(
                        pl.lit(-1).cast(base.schema["play_id"]).alias("play_id")
                    ),
                ],
                how="diagonal_relaxed",
            )
        return base

    with pytest.raises(LeakageError, match="row"):
        check_delete_the_future(unstable, {"pbp": pbp}, cut=100)


# --------------------------------------------------------------------------- #
# check_no_market
# --------------------------------------------------------------------------- #


def test_check_no_market_passes_a_clean_frame(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    df = assemble_dataset([2024], pbp=pbp, schedules=schedules)
    assert check_no_market(df) == []


def test_check_no_market_rejects_a_dirty_frame(pbp_fixture: pl.DataFrame) -> None:
    with pytest.raises(MarketColumnLeak):
        check_no_market(pbp_fixture, context="test")


# --------------------------------------------------------------------------- #
# check_no_trivial_leak
# --------------------------------------------------------------------------- #


def _toy_dataset(extra: dict[str, list[float]] | None = None) -> pl.DataFrame:
    """Four games, two home wins, a plausible feature and whatever ``extra`` adds."""
    data: dict[str, list] = {
        "meta__game_id": [g for g in ("a", "b", "c", "d") for _ in range(4)],
        "meta__play_id": [1, 2, 3, 4] * 4,
        LABEL_COLUMN: [1.0] * 4 + [0.0] * 4 + [1.0] * 4 + [0.0] * 4,
        "bi__score_diff_home": [0, 3, 7, 10, 0, -3, -7, -1, 0, 7, 3, 4, 0, 0, -3, -6],
    }
    data.update(extra or {})
    return pl.DataFrame(data)


def test_a_reasonable_feature_passes() -> None:
    report = check_no_trivial_leak(_toy_dataset())
    assert "bi__score_diff_home" in report.checked
    assert report.suspects == ()
    assert report


def test_a_copy_of_the_label_is_caught() -> None:
    leak = {"bi__cheat": [1.0] * 4 + [0.0] * 4 + [1.0] * 4 + [0.0] * 4}
    with pytest.raises(LeakageError, match="bi__cheat"):
        check_no_trivial_leak(_toy_dataset(leak))


def test_an_end_of_game_giveaway_is_caught_per_game() -> None:
    """Diluted pooled, obvious on the last play of each game."""
    leak = {"bi__final_margin": [0, 0, 0, 20, 0, 0, 0, -20, 0, 0, 0, 20, 0, 0, 0, -20]}
    report = check_no_trivial_leak(_toy_dataset(leak), raise_on_leak=False)
    assert [s.column for s in report.suspects] == ["bi__final_margin"]
    assert report.suspects[0].scope == "per_game"
    assert not report


def test_non_numeric_and_constant_columns_are_skipped() -> None:
    frame = _toy_dataset({"bi__constant": [1.0] * 16}).with_columns(
        pl.lit("dome").alias("bi__roof")
    )
    report = check_no_trivial_leak(frame)
    assert report.skipped["bi__roof"] == "non-numeric"
    assert "bi__constant" in report.skipped


def test_threshold_is_configurable() -> None:
    report = check_no_trivial_leak(_toy_dataset(), threshold=0.1, raise_on_leak=False)
    assert [s.column for s in report.suspects]


def test_a_missing_label_raises() -> None:
    with pytest.raises(KeyError):
        check_no_trivial_leak(pl.DataFrame({"bi__x": [1.0]}))


def test_the_assembled_dataset_has_no_trivial_leak(
    pbp_fixture: pl.DataFrame, schedules_fixture: pl.DataFrame
) -> None:
    df = assemble_dataset(pbp=pbp_fixture, schedules=schedules_fixture)
    assert check_no_trivial_leak(df).suspects == ()
