"""Replay feed built from a one-game slice of nflverse play-by-play."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from nfl.data.live.replay import HistoricalReplayFeed, _yards_to_endzone

GAME_ID = "2024_01_BAL_KC"


@pytest.fixture
def feed(pbp_path: Path, schedules_path: Path) -> HistoricalReplayFeed:
    return HistoricalReplayFeed(pbp=pbp_path, schedules=schedules_path)


def test_full_replay_is_the_final_state(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID)
    assert snapshot.source == "replay"
    assert snapshot.state == "post"
    assert snapshot.completed is True
    assert (snapshot.home_team, snapshot.away_team) == ("KC", "BAL")
    assert (snapshot.home_score, snapshot.away_score) == (27, 20)
    assert snapshot.game_id == GAME_ID
    assert snapshot.espn_event_id == "401671789"
    assert snapshot.season == 2024
    assert snapshot.week == 1
    # A finished game has no live situation.
    assert snapshot.possession is None
    assert snapshot.down is None


def test_rewind_to_a_clock(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID, at="Q4 05:00")
    assert (snapshot.period, snapshot.clock_seconds) == (4, 300)
    assert (snapshot.home_score, snapshot.away_score) == (27, 17)
    assert snapshot.possession == "BAL"
    assert snapshot.posteam_is_home is False
    assert (snapshot.down, snapshot.distance, snapshot.yards_to_endzone) == (4, 4, 14)
    assert snapshot.is_red_zone is True
    assert snapshot.state == "in"
    # nflverse carries per-play timeouts, so the rewound value is exact.
    assert (snapshot.home_timeouts, snapshot.away_timeouts) == (3, 1)
    assert snapshot.timeouts_exact is True
    assert snapshot.game_seconds_remaining == 300


def test_plays_carry_post_play_scores_and_pre_snap_downs(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID)
    field_goal = next(p for p in snapshot.plays if p.play_id == "3684")
    assert field_goal.start_down == 4
    assert field_goal.start_distance == 4
    assert field_goal.start_yards_to_endzone == 14
    assert field_goal.scoring_play is True
    # Score AFTER the kick.
    assert (field_goal.home_score, field_goal.away_score) == (27, 20)
    assert field_goal.wallclock == datetime(2024, 9, 6, 3, 28, 20, tzinfo=UTC)


def test_kickoff_state(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID, at="Q1 15:00")
    assert (snapshot.home_score, snapshot.away_score) == (0, 0)
    assert snapshot.play_count == 0
    assert snapshot.drives == []
    assert snapshot.state == "in"
    # nflverse's game-start row has no posteam: nobody has the ball yet.
    assert snapshot.possession is None


def test_drives_are_grouped_and_labelled(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID)
    assert len(snapshot.drives) == 20
    assert [d.drive_number for d in snapshot.drives] == list(range(1, 21))
    scoring = [d for d in snapshot.drives if d.is_score]
    assert scoring
    assert all(d.result for d in snapshot.drives[1:])
    assert snapshot.drives[1].description is not None


def test_venue_and_weather_come_from_the_pbp_columns(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID)
    assert snapshot.venue is not None
    assert snapshot.venue.venue_id == "KAN00"
    assert snapshot.venue.roof == "outdoors"
    assert snapshot.venue.indoor is False
    assert snapshot.venue.grass is True
    assert snapshot.weather is not None
    assert snapshot.weather.temperature_f == 67
    assert snapshot.weather.wind_mph == 8


def test_records_are_as_of_the_game_week(feed: HistoricalReplayFeed) -> None:
    snapshot = feed.snapshot(GAME_ID)
    assert snapshot.home_record is not None
    assert snapshot.home_record.overall == "0-0"  # week 1: nothing has happened yet
    assert snapshot.away_record is not None
    assert snapshot.away_record.overall == "0-0"


def test_replay_has_no_in_game_injuries(feed: HistoricalReplayFeed) -> None:
    """nflverse pbp carries none; empty rather than fabricated."""
    assert feed.snapshot(GAME_ID).injuries == []


def test_list_games_reads_the_schedule(feed: HistoricalReplayFeed) -> None:
    games = feed.list_games(date(2024, 9, 5))
    assert [g.nflverse_game_id for g in games] == [GAME_ID]
    assert games[0].espn_event_id == "401671789"
    assert games[0].completed is True


def test_unknown_game_raises(feed: HistoricalReplayFeed) -> None:
    with pytest.raises(KeyError):
        feed.snapshot("2024_01_XXX_YYY")


def test_accepts_an_in_memory_frame(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    snapshot = HistoricalReplayFeed(pbp=pbp, schedules=schedules).snapshot(GAME_ID, at="Q2 02:00")
    assert snapshot.period == 2
    assert snapshot.clock_seconds == 120


def test_accepts_a_directory_of_parquet(pbp_path: Path, schedules_path: Path) -> None:
    """The directory holds schedules too; only play-by-play files may be read."""
    feed = HistoricalReplayFeed(pbp=pbp_path.parent, schedules=schedules_path)
    assert feed.snapshot(GAME_ID).play_count == 178


@pytest.mark.parametrize(
    ("yard_line", "posteam", "expected"),
    [
        ("KC 32", "KC", 68),
        ("KC 32", "BAL", 32),
        ("50", "KC", 50),
        (None, "KC", None),
        ("KC 32", None, None),
        ("nonsense", "KC", None),
    ],
)
def test_yards_to_endzone_parsing(
    yard_line: str | None, posteam: str | None, expected: int | None
) -> None:
    assert _yards_to_endzone(yard_line, posteam) == expected
