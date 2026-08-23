"""Closing-line reader tests. Pure filesystem, no network."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.data.market import closing_lines
from nfl.data.schemas import Market

# Real nflverse row: 2024_01_BAL_KC, KC home, spread_line 3, home ML -148.
SCHEDULE_ROWS = {
    "game_id": ["2024_01_BAL_KC", "2024_02_XX_YY", "2024_03_NO_LINE"],
    "season": [2024, 2024, 2024],
    "week": [1, 2, 3],
    "gameday": ["2024-09-05", "2024-09-12", "2024-09-19"],
    "gametime": ["20:20", "13:00", None],
    "away_team": ["BAL", "XX", "AA"],
    "home_team": ["KC", "YY", "BB"],
    "away_moneyline": [124, -110, None],
    "home_moneyline": [-148, -110, None],
    "spread_line": [3.0, 0.0, None],
    "total_line": [46.0, 41.0, None],
}


@pytest.fixture
def schedule_parquet(tmp_path):
    path = tmp_path / "nflverse" / "market"
    path.mkdir(parents=True)
    target = path / "schedules.parquet"
    pl.DataFrame(SCHEDULE_ROWS).write_parquet(target)
    return target


# --------------------------------------------------------------------------- #
# Odds maths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("odds", "expected"),
    [(-148, 0.5968), (124, 0.4464), (-110, 0.5238), (100, 0.5), (None, None)],
)
def test_american_to_probability(odds, expected):
    got = closing_lines.american_to_probability(odds)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-4)


def test_devig_sums_to_one():
    home, away = closing_lines.devig(0.5968, 0.4464)
    assert home + away == pytest.approx(1.0)
    assert home > away


def test_devig_handles_one_missing_side():
    home, away = closing_lines.devig(0.6, None)
    assert (home, away) == (0.6, pytest.approx(0.4))
    assert closing_lines.devig(None, None) == (None, None)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_find_schedule_file(tmp_path, schedule_parquet):
    assert closing_lines.find_schedule_file(tmp_path) == schedule_parquet


def test_find_schedule_file_returns_none_when_absent(tmp_path):
    assert closing_lines.find_schedule_file(tmp_path / "nothing") is None


def test_load_schedule_raises_when_nothing_on_disk(tmp_path):
    with pytest.raises(closing_lines.ClosingLinesNotFound):
        closing_lines.load_schedule(root=tmp_path)


def test_load_schedule_rejects_non_schedule_file(tmp_path):
    path = tmp_path / "nflverse" / "market"
    path.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(path / "schedules.parquet")
    with pytest.raises(KeyError):
        closing_lines.load_schedule(root=tmp_path)


def test_load_schedule_filters_seasons(schedule_parquet):
    frame = closing_lines.load_schedule(schedule_parquet, seasons=[2023])
    assert frame.height == 0


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def test_market_rows_use_nflverse_spread_convention(schedule_parquet):
    rows = closing_lines.market_rows(closing_lines.load_schedule(schedule_parquet))
    by_id = {row.game_id: row for row in rows}

    kc = by_id["2024_01_BAL_KC"]
    assert isinstance(kc, Market)
    assert kc.market_only is True
    assert kc.book == "consensus"
    # spread_line is the home team's expected margin; KC favoured by 3.
    assert kc.spread_home == 3.0
    assert kc.spread_close_home == 3.0
    assert kc.spread_open_home is None
    assert kc.total_close == 46.0
    assert kc.moneyline_home == -148


def test_market_rows_devig_moneylines(schedule_parquet):
    rows = closing_lines.market_rows(closing_lines.load_schedule(schedule_parquet))
    kc = next(row for row in rows if row.game_id == "2024_01_BAL_KC")
    assert kc.implied_prob_home == pytest.approx(0.5721, abs=1e-3)
    assert kc.implied_prob_home + kc.implied_prob_away == pytest.approx(1.0)


def test_rows_without_any_line_are_skipped(schedule_parquet):
    rows = closing_lines.market_rows(closing_lines.load_schedule(schedule_parquet))
    assert {row.game_id for row in rows} == {"2024_01_BAL_KC", "2024_02_XX_YY"}


def test_observed_at_is_kickoff(schedule_parquet):
    rows = closing_lines.market_rows(closing_lines.load_schedule(schedule_parquet))
    kc = next(row for row in rows if row.game_id == "2024_01_BAL_KC")
    assert (kc.observed_at_utc.year, kc.observed_at_utc.month, kc.observed_at_utc.day) == (
        2024,
        9,
        5,
    )
    assert kc.observed_at_utc.hour == 20


def test_market_collapsed_is_false_for_normal_lines(schedule_parquet):
    rows = closing_lines.market_rows(closing_lines.load_schedule(schedule_parquet))
    assert all(row.market_collapsed is False for row in rows)


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def test_ingest_writes_one_parquet_per_season(tmp_path, schedule_parquet):
    paths = closing_lines.ingest_closing_lines([2024], path=schedule_parquet, root=tmp_path)
    assert len(paths) == 1
    assert paths[0].parent == tmp_path / "market" / "closing_lines" / "season=2024"

    frame = pl.read_parquet(paths[0])
    assert frame.height == 2
    assert frame["market_only"].to_list() == [True, True]


def test_ingest_is_idempotent(tmp_path, schedule_parquet):
    first = closing_lines.ingest_closing_lines([2024], path=schedule_parquet, root=tmp_path)[0]
    mtime = first.stat().st_mtime_ns
    again = closing_lines.ingest_closing_lines([2024], path=schedule_parquet, root=tmp_path)[0]
    assert again.stat().st_mtime_ns == mtime


def test_ingest_force_rewrites(tmp_path, schedule_parquet):
    closing_lines.ingest_closing_lines([2024], path=schedule_parquet, root=tmp_path)
    path = closing_lines.ingest_closing_lines(
        [2024], path=schedule_parquet, root=tmp_path, force=True
    )[0]
    assert pl.read_parquet(path).height == 2


def test_csv_fallback(tmp_path):
    directory = tmp_path / "nflverse" / "market"
    directory.mkdir(parents=True)
    csv_path = directory / "games.csv"
    pl.DataFrame(SCHEDULE_ROWS).write_csv(csv_path)
    frame = closing_lines.load_schedule(csv_path)
    assert frame.height == 3
