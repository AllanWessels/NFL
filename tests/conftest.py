"""Shared offline fixtures for the data-layer tests.

Nothing here touches the network. The nflverse loaders are replaced with
functions that slice the committed parquet fixtures in ``tests/fixtures/``, so
the tests exercise the real ingest code path — season filtering, the market
split, provenance stamping, partition writing — against known rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import polars as pl
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def schedules_fixture() -> pl.DataFrame:
    """The 9-game, 46-column schedules fixture, betting columns included."""
    return pl.read_parquet(FIXTURE_DIR / "nflverse_schedules.parquet")


@pytest.fixture(scope="session")
def pbp_fixture() -> pl.DataFrame:
    """An 8-play play-by-play fixture carrying every blocked column."""
    return pl.read_parquet(FIXTURE_DIR / "nflverse_pbp.parquet")


@pytest.fixture
def offline_nflverse(
    monkeypatch: pytest.MonkeyPatch,
    schedules_fixture: pl.DataFrame,
    pbp_fixture: pl.DataFrame,
) -> Iterator[dict[str, list[list[int]]]]:
    """Replace the nflreadpy loaders with fixture slices; record every call.

    Yields a ``{loader_name: [seasons, ...]}`` log so a test can assert *how*
    the vendor was called — that a whole-history asset was fetched once rather
    than once per season, for instance.
    """
    import nflreadpy

    calls: dict[str, list[list[int]]] = {}

    def _slice(frame: pl.DataFrame, name: str) -> Callable[..., pl.DataFrame]:
        def loader(seasons: object = None, **kwargs: object) -> pl.DataFrame:
            wanted = _as_season_list(seasons)
            calls.setdefault(name, []).append(wanted)
            if wanted is None:
                return frame
            out = frame.filter(pl.col("season").is_in(wanted))
            if "stat_type" in kwargs:
                out = out.with_columns(pl.lit(str(kwargs["stat_type"])).alias("_variant"))
            return out

        return loader

    monkeypatch.setattr(nflreadpy, "load_pbp", _slice(pbp_fixture, "load_pbp"))
    monkeypatch.setattr(nflreadpy, "load_schedules", _slice(schedules_fixture, "load_schedules"))
    for name in (
        "load_rosters_weekly",
        "load_injuries",
        "load_depth_charts",
        "load_snap_counts",
        "load_participation",
        "load_nextgen_stats",
        "load_pfr_advstats",
    ):
        monkeypatch.setattr(nflreadpy, name, _slice(_stub_weekly_frame(), name))

    # The real one writes into DATA_DIR and mutates process-global config.
    monkeypatch.setattr("nfl.data.nflverse.configure_cache", lambda cache_dir=None: Path("."))
    yield calls


def _as_season_list(seasons: object) -> list[int] | None:
    if seasons is None or seasons is True:
        return None
    if isinstance(seasons, int):
        return [seasons]
    return [int(s) for s in seasons]  # type: ignore[union-attr]


def _stub_weekly_frame() -> pl.DataFrame:
    """A minimal ``(season, week, team)``-keyed frame standing in for the weekly tables."""
    return pl.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "week": [1, 2, 1, 2],
            "team": ["KC", "KC", "KC", "KC"],
            "player_id": ["00-0033873"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
