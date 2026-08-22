"""ADR 0001 enforcement: no market column ever reaches a feature frame."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.data.market_blocklist import (
    MARKET_COLUMNS,
    MARKET_DERIVED_WP_COLUMNS,
    MARKET_QUOTE_COLUMNS,
    THIRD_PARTY_WP_COLUMNS,
    MarketColumnLeak,
    assert_no_market_columns,
    drop_market_columns,
    market_columns_in,
)
from nfl.data.nflverse import _split_market

# The eight betting columns nflverse ships in `schedules`; `spread_line` and
# `total_line` are additionally denormalised onto every play-by-play row.
_VENDOR_QUOTES = {
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
}


def test_no_market_columns_in_feature_sets(pbp_fixture: pl.DataFrame) -> None:
    """The guard the whole ADR rests on, exercised end to end.

    A raw play-by-play frame is *supposed* to fail the check — it carries the
    spread, the total and the vegas win-probability columns. The frame a feature
    builder is allowed to hand a model is what survives
    :func:`drop_market_columns`, and that must pass.
    """
    with pytest.raises(MarketColumnLeak):
        assert_no_market_columns(pbp_fixture, context="raw pbp")

    feature_set = drop_market_columns(pbp_fixture)
    assert_no_market_columns(feature_set, context="feature set")

    # The check is disjointness, not merely "no exception": spell it out.
    assert set(feature_set.columns).isdisjoint(MARKET_COLUMNS)

    # And the drop is surgical — the state columns a WP model actually needs
    # are all still there.
    for kept in ("game_id", "score_differential", "game_seconds_remaining", "yardline_100", "epa"):
        assert kept in feature_set.columns


def test_blocklist_covers_every_vendor_betting_column() -> None:
    assert _VENDOR_QUOTES <= MARKET_COLUMNS
    assert MARKET_QUOTE_COLUMNS == _VENDOR_QUOTES


def test_blocklist_covers_spread_derived_win_probability() -> None:
    """The subtle leak: `vegas_*` is the closing line wearing a lab coat."""
    assert {"vegas_wp", "vegas_home_wp", "vegas_wpa", "vegas_home_wpa"} <= MARKET_DERIVED_WP_COLUMNS
    assert MARKET_DERIVED_WP_COLUMNS <= MARKET_COLUMNS


def test_blocklist_covers_third_party_win_probability() -> None:
    """CLAUDE.md rule 2: another model's estimate of our target is not a feature."""
    assert {"wp", "wpa", "home_wp", "away_wp", "def_wp"} <= THIRD_PARTY_WP_COLUMNS
    assert THIRD_PARTY_WP_COLUMNS <= MARKET_COLUMNS


def test_pattern_rule_catches_columns_not_yet_in_the_list() -> None:
    """A vendor rename must fail closed, not sail through."""
    assert market_columns_in(["vegas_home_wp_post"]) == ["vegas_home_wp_post"]
    assert market_columns_in(["closing_moneyline_home"]) == ["closing_moneyline_home"]
    assert market_columns_in(["alt_spread_odds"]) == ["alt_spread_odds"]
    assert market_columns_in(["score_differential", "posteam"]) == []


def test_leak_error_names_the_offenders() -> None:
    with pytest.raises(MarketColumnLeak) as excinfo:
        assert_no_market_columns(["down", "spread_line", "wp"], context="unit test")
    assert excinfo.value.offenders == ["spread_line", "wp"]
    assert "0001-no-market-inputs" in str(excinfo.value)
    assert "unit test" in str(excinfo.value)


def test_accepts_lazyframes_and_mappings(pbp_fixture: pl.DataFrame) -> None:
    with pytest.raises(MarketColumnLeak):
        assert_no_market_columns(pbp_fixture.lazy())
    assert_no_market_columns(drop_market_columns(pbp_fixture).lazy())
    assert_no_market_columns({"down": [1], "ydstogo": [10]})


def test_ingested_schedules_carry_no_quotes_and_market_table_keeps_them(
    schedules_fixture: pl.DataFrame,
) -> None:
    """The split is physical: the quotes leave `schedules` and land in `market`."""
    market = _split_market(schedules_fixture)

    assert set(market.columns) >= _VENDOR_QUOTES
    assert market["market_only"].all()
    assert market["line_type"].unique().to_list() == ["close"]

    row = market.filter(pl.col("game_id") == "2023_01_DEN_KC").row(0, named=True)
    assert row["spread_line"] == -7.0
    assert row["total_line"] == 53.5
    assert row["home_moneyline"] == -340
