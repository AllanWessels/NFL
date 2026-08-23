"""The fallback play-level builder: pre-snap only, no market columns, no future."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.data.market_blocklist import market_columns_in
from nfl.features.baseline_inputs import FEATURE_DOCS, PREFIX, build_baseline_inputs
from nfl.features.leakage import check_delete_the_future


def test_one_row_per_play(pbp: pl.DataFrame) -> None:
    out = build_baseline_inputs(pbp)
    assert out.height == pbp.height
    assert not out.select(["game_id", "play_id"]).is_duplicated().any()


def test_every_feature_is_prefixed_and_documented(pbp: pl.DataFrame) -> None:
    out = build_baseline_inputs(pbp)
    features = [c for c in out.columns if c not in ("game_id", "play_id")]
    assert features
    assert all(c.startswith(PREFIX) for c in features)
    assert set(features) == set(FEATURE_DOCS)


def test_no_market_columns_even_from_a_dirty_pbp(pbp_fixture: pl.DataFrame) -> None:
    assert market_columns_in(pbp_fixture)  # the fixture really is dirty
    assert market_columns_in(build_baseline_inputs(pbp_fixture)) == []


def test_score_is_pre_snap_not_post_play(pbp: pl.DataFrame) -> None:
    """The touchdown's points must not be on the touchdown's row."""
    frame = pbp.sort(["game_id", "play_id"]).with_row_index("i")
    scoring = frame.filter(pl.col("sp") == 1).row(0, named=True)
    before = frame.filter(pl.col("i") == scoring["i"] - 1).row(0, named=True)
    out = build_baseline_inputs(pbp).filter(pl.col("play_id") == int(scoring["play_id"]))
    row = out.row(0, named=True)

    assert row["bi__home_score"] == before["total_home_score"]
    assert row["bi__away_score"] == before["total_away_score"]
    # And the scoring play's own points are visibly not counted yet.
    assert scoring["total_away_score"] != before["total_away_score"]


def test_first_play_of_a_game_starts_at_nil(pbp: pl.DataFrame) -> None:
    out = build_baseline_inputs(pbp).sort(["game_id", "play_id"])
    first = out.group_by("game_id").first()
    assert first.get_column("bi__home_score").to_list() == [0.0] * first.height
    assert first.get_column("bi__away_score").to_list() == [0.0] * first.height


def test_possession_flag_matches_the_home_team(pbp: pl.DataFrame) -> None:
    joined = build_baseline_inputs(pbp).join(
        pbp.select("game_id", pl.col("play_id").cast(pl.Int64), "posteam", "home_team"),
        on=["game_id", "play_id"],
    )
    with_ball = joined.filter(pl.col("posteam").is_not_null())
    expected = (with_ball.get_column("posteam") == with_ball.get_column("home_team")).cast(pl.Int8)
    assert with_ball.get_column("bi__posteam_is_home").to_list() == expected.to_list()


def test_timeouts_are_derived_from_whichever_pair_nflverse_ships(
    pbp: pl.DataFrame, pbp_fixture: pl.DataFrame
) -> None:
    # The 2024 fixture carries home_/away_; the older one carries posteam_/defteam_.
    assert "home_timeouts_remaining" in pbp.columns
    assert "posteam_timeouts_remaining" in pbp_fixture.columns

    from_side = build_baseline_inputs(pbp).drop_nulls("bi__posteam_timeouts_remaining")
    from_poss = build_baseline_inputs(pbp_fixture).drop_nulls("bi__home_timeouts_remaining")
    for out in (from_side, from_poss):
        assert out.height > 0
        for col in (
            "bi__home_timeouts_remaining",
            "bi__away_timeouts_remaining",
            "bi__posteam_timeouts_remaining",
            "bi__defteam_timeouts_remaining",
        ):
            assert out.get_column(col).is_between(0, 3).all()


def test_delete_the_future(pbp: pl.DataFrame) -> None:
    cut = int(pbp.get_column("play_id").sort()[pbp.height // 2])
    check = check_delete_the_future(build_baseline_inputs, {"pbp": pbp}, cut, order_col="play_id")
    assert check.rows_compared > 0
    assert check.columns_compared


def test_missing_keys_raise() -> None:
    with pytest.raises(KeyError, match="play_id"):
        build_baseline_inputs(pl.DataFrame({"game_id": ["g"]}))
