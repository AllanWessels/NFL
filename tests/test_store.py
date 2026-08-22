"""The DuckDB catalog: views over parquet globs, never copies of the data."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl.data.coach_records import ingest_coach_records
from nfl.data.nflverse import ingest_nflverse
from nfl.data.store import connect, discover_tables, open_db, refresh_views, table_names


@pytest.fixture
def lake(tmp_path: Path, offline_nflverse: dict) -> tuple[Path, Path]:
    """A miniature data lake: two raw nflverse tables plus one processed table."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    ingest_nflverse([2023, 2024], ["pbp", "schedules", "market"], root=raw)
    ingest_coach_records([2023, 2024], raw_root=raw, processed_root=processed)
    return raw, processed


def test_open_db_creates_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "nfl.duckdb"
    con = open_db(path)
    try:
        assert path.exists()
    finally:
        con.close()


def test_discover_names_raw_and_processed_tables(lake: tuple[Path, Path]) -> None:
    raw, processed = lake
    specs = {spec.view: spec for spec in discover_tables(raw, processed)}
    assert {"nflverse_pbp", "nflverse_schedules", "nflverse_market"} <= set(specs)
    assert "coach_records" in specs
    assert specs["nflverse_pbp"].layer == "raw"
    assert specs["coach_records"].layer == "processed"


def test_refresh_views_registers_every_table(lake: tuple[Path, Path]) -> None:
    raw, processed = lake
    with connect(":memory:") as con:
        views = refresh_views(con, raw, processed)
        assert views == sorted(views)
        assert set(views) == {
            "nflverse_pbp",
            "nflverse_schedules",
            "nflverse_market",
            "coach_records",
        }
        assert set(table_names(con)) == set(views)


def test_views_read_every_season_partition(lake: tuple[Path, Path]) -> None:
    raw, processed = lake
    with connect(":memory:") as con:
        refresh_views(con, raw, processed)
        seasons = con.execute("SELECT DISTINCT season FROM nflverse_pbp ORDER BY season").fetchall()
        assert [row[0] for row in seasons] == [2023, 2024]

        total = con.execute("SELECT count(*) FROM nflverse_pbp").fetchone()
        on_disk = pl.read_parquet(str(raw / "nflverse" / "pbp" / "**" / "*.parquet")).height
        assert total is not None and total[0] == on_disk


def test_hive_partition_filter_selects_one_season(lake: tuple[Path, Path]) -> None:
    """`season` lives in the path *and* the file; both must agree."""
    raw, processed = lake
    with connect(":memory:") as con:
        refresh_views(con, raw, processed)
        rows = con.execute(
            "SELECT DISTINCT season FROM nflverse_pbp WHERE season = 2024"
        ).fetchall()
        assert rows == [(2024,)]


def test_market_view_is_separate_from_schedules(lake: tuple[Path, Path]) -> None:
    """ADR 0001 survives the round trip through the catalog."""
    raw, processed = lake
    with connect(":memory:") as con:
        refresh_views(con, raw, processed)
        schedule_cols = {row[0] for row in con.execute("DESCRIBE nflverse_schedules").fetchall()}
        market_cols = {row[0] for row in con.execute("DESCRIBE nflverse_market").fetchall()}
        assert "spread_line" not in schedule_cols
        assert "spread_line" in market_cols
        assert "market_only" in market_cols


def test_refresh_is_idempotent_and_drops_stale_views(lake: tuple[Path, Path]) -> None:
    raw, processed = lake
    with connect(":memory:") as con:
        first = refresh_views(con, raw, processed)
        assert refresh_views(con, raw, processed) == first

        import shutil

        shutil.rmtree(raw / "nflverse" / "market")
        after = refresh_views(con, raw, processed)
        assert "nflverse_market" not in after
        assert "nflverse_market" not in table_names(con)


def test_empty_lake_registers_nothing(tmp_path: Path) -> None:
    with connect(":memory:") as con:
        assert refresh_views(con, tmp_path / "raw", tmp_path / "processed") == []


def test_directory_without_parquet_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "raw" / "nflverse" / "half_written").mkdir(parents=True)
    with connect(":memory:") as con:
        assert refresh_views(con, tmp_path / "raw", tmp_path / "processed") == []
