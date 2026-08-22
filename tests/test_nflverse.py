"""nflverse ingest: season parsing, the market split, and partition writing.

Every test runs offline against the committed parquet fixtures — see
``tests/conftest.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from nfl.data.market_blocklist import MARKET_QUOTE_COLUMNS, assert_no_market_columns
from nfl.data.nflverse import (
    SOURCE,
    TABLE_NAMES,
    TABLES,
    configure_cache,
    fetch_nflverse,
    ingest_nflverse,
    parse_seasons,
    season_partition_path,
)

# --------------------------------------------------------------------------- #
# Season parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2024", [2024]),
        ("2010-2013", [2010, 2011, 2012, 2013]),
        ("2016,2018-2020", [2016, 2018, 2019, 2020]),
        (" 2023 , 2023 ", [2023]),
    ],
)
def test_parse_seasons(spec: str, expected: list[int]) -> None:
    assert parse_seasons(spec) == expected


@pytest.mark.parametrize("spec", ["2020-2015", "", "2005", "1999-2012"])
def test_parse_seasons_rejects_bad_input(spec: str) -> None:
    """Seasons before 2010 are out of scope (research §3: the coverage gaps)."""
    with pytest.raises(ValueError):
        parse_seasons(spec)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_covers_the_required_tables() -> None:
    required = {
        "pbp",
        "schedules",
        "market",
        "weekly_rosters",
        "injuries",
        "depth_charts",
        "snap_counts",
        "ngs",
        "pfr_advstats",
        "participation",
        "qbr",
    }
    assert required <= set(TABLE_NAMES)


@pytest.mark.parametrize(
    ("table", "min_season"),
    [("ngs", 2016), ("pfr_advstats", 2018), ("participation", 2016), ("snap_counts", 2012)],
)
def test_coverage_floors_match_upstream(table: str, min_season: int) -> None:
    """Requesting a season the vendor never published must not silently write junk."""
    assert TABLES[table].min_season == min_season


def test_configure_cache_sets_filesystem_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nflreadpy defaults to an in-memory cache — 300 MB re-downloaded per process."""
    from nflreadpy.config import CacheMode, get_config

    # Seed env vars and the process-global config object through monkeypatch, so
    # its teardown restores both and this test cannot leak into the others.
    monkeypatch.setenv("NFLREADPY_CACHE", "memory")
    monkeypatch.setenv("NFLREADPY_CACHE_DIR", "")
    live = get_config()
    monkeypatch.setattr(live, "cache_mode", live.cache_mode)
    monkeypatch.setattr(live, "cache_dir", live.cache_dir)

    target = configure_cache(tmp_path / "cache")
    assert target.is_dir()
    assert os.environ["NFLREADPY_CACHE"] == "filesystem"
    assert os.environ["NFLREADPY_CACHE_DIR"] == str(target)

    config = get_config()
    assert config.cache_mode is CacheMode.FILESYSTEM
    assert Path(config.cache_dir) == target


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def test_fetch_filters_seasons_and_stamps_provenance(offline_nflverse: dict) -> None:
    df = fetch_nflverse("pbp", [2023])
    assert df["season"].unique().to_list() == [2023]
    assert df["source"].unique().to_list() == ["nflverse.pbp"]
    assert df["ingested_at"].null_count() == 0
    assert "schema_version" in df.columns


def test_fetch_below_coverage_floor_returns_empty(offline_nflverse: dict) -> None:
    assert fetch_nflverse("ngs", [2012]).height == 0
    assert "load_nextgen_stats" not in offline_nflverse


def test_fetch_schedules_drops_the_betting_columns(offline_nflverse: dict) -> None:
    schedules = fetch_nflverse("schedules", [2023, 2024])
    assert_no_market_columns(schedules, context="ingested schedules")
    # The crosswalk, coach, roof and venue columns must survive the drop.
    for kept in ("espn", "home_coach", "away_coach", "roof", "stadium_id", "stadium", "referee"):
        assert kept in schedules.columns


def test_fetch_market_keeps_the_betting_columns(offline_nflverse: dict) -> None:
    market = fetch_nflverse("market", [2023, 2024])
    assert set(market.columns) >= MARKET_QUOTE_COLUMNS
    assert market["market_only"].all()
    assert market["source"].unique().to_list() == ["market"]


def test_market_observed_at_is_kickoff_in_utc(offline_nflverse: dict) -> None:
    """Closing lines have no timestamp of their own; kickoff is what "closing" means."""
    market = fetch_nflverse("market", [2023])
    row = market.filter(pl.col("game_id") == "2023_01_DEN_KC").row(0, named=True)
    # 16:25 US/Eastern on 2023-09-10 is EDT (UTC-4) -> 20:25Z.
    assert row["observed_at_utc"].hour == 20
    assert row["observed_at_utc"].minute == 25


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def test_ingest_writes_hive_partitions(tmp_path: Path, offline_nflverse: dict) -> None:
    root = ingest_nflverse([2023, 2024], ["pbp", "schedules"], root=tmp_path)
    assert root == tmp_path / SOURCE

    for season in (2023, 2024):
        path = season_partition_path("pbp", season, tmp_path)
        assert path.exists()
        assert path.parent.name == f"season={season}"
        part = pl.read_parquet(path)
        assert part["season"].unique().to_list() == [season]

    # And the whole table reads back as one frame through the glob.
    everything = pl.read_parquet(str(tmp_path / SOURCE / "pbp" / "**" / "*.parquet"))
    assert sorted(everything["season"].unique().to_list()) == [2023, 2024]


def test_ingest_is_idempotent_and_force_overwrites(tmp_path: Path, offline_nflverse: dict) -> None:
    from nfl.data.nflverse import IngestResult

    first: list[IngestResult] = []
    ingest_nflverse([2023], ["pbp"], root=tmp_path, results=first)
    assert [r.written for r in first] == [True]

    second: list[IngestResult] = []
    ingest_nflverse([2023], ["pbp"], root=tmp_path, results=second)
    assert [r.written for r in second] == [False]
    assert second[0].reason == "exists"

    third: list[IngestResult] = []
    ingest_nflverse([2023], ["pbp"], root=tmp_path, force=True, results=third)
    assert [r.written for r in third] == [True]


def test_ingest_skips_seasons_before_coverage_floor(tmp_path: Path, offline_nflverse: dict) -> None:
    from nfl.data.nflverse import IngestResult

    results: list[IngestResult] = []
    ingest_nflverse([2010, 2016], ["ngs"], root=tmp_path, results=results)
    by_season = {r.season: r for r in results}
    assert by_season[2010].written is False
    assert "not available before 2016" in by_season[2010].reason
    assert not season_partition_path("ngs", 2010, tmp_path).exists()


def test_whole_history_assets_are_fetched_once(tmp_path: Path, offline_nflverse: dict) -> None:
    """`games.parquet` is one file for all seasons — don't refetch it per season."""
    ingest_nflverse([2023, 2024], ["schedules"], root=tmp_path)
    assert offline_nflverse["load_schedules"] == [[2023, 2024]]


def test_per_season_assets_are_fetched_per_season(tmp_path: Path, offline_nflverse: dict) -> None:
    """Play-by-play is ~20 MB/season; fetch one at a time to cap peak memory."""
    ingest_nflverse([2023, 2024], ["pbp"], root=tmp_path)
    assert offline_nflverse["load_pbp"] == [[2023], [2024]]


def test_ingested_schedules_partition_is_market_free(
    tmp_path: Path, offline_nflverse: dict
) -> None:
    """The physical guarantee: a `select *` off disk cannot pick up a line."""
    ingest_nflverse([2023], ["schedules", "market"], root=tmp_path)

    schedules = pl.read_parquet(season_partition_path("schedules", 2023, tmp_path))
    assert_no_market_columns(schedules, context="schedules partition")

    market = pl.read_parquet(season_partition_path("market", 2023, tmp_path))
    assert set(market.columns) >= MARKET_QUOTE_COLUMNS


def test_unknown_table_is_rejected(tmp_path: Path, offline_nflverse: dict) -> None:
    with pytest.raises(KeyError, match="unknown nflverse table"):
        ingest_nflverse([2023], ["not_a_table"], root=tmp_path)
