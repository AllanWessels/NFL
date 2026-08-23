"""Feed-agnostic pieces: clock maths, ``--at`` parsing, the crosswalk, truncation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl.data.live.base import (
    AtTime,
    LiveDrive,
    LivePlay,
    LiveSnapshot,
    clock_to_seconds,
    espn_team_abbr_from_id,
    espn_to_nflverse_team,
    game_seconds_remaining,
    half_seconds_remaining,
    nflverse_to_espn_team,
    parse_at,
    resolve_espn_event_id,
    seconds_to_clock,
    team_record_as_of,
    truncate_snapshot,
)

# --------------------------------------------------------------------------- #
# Team identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("espn", "nflverse"),
    [("WSH", "WAS"), ("LAR", "LA"), ("KC", "KC"), ("LV", "LV"), ("JAX", "JAX")],
)
def test_team_abbreviations_round_trip(espn: str, nflverse: str) -> None:
    assert espn_to_nflverse_team(espn) == nflverse
    assert nflverse_to_espn_team(nflverse) == espn


def test_team_id_lookup_applies_the_rename() -> None:
    assert espn_team_abbr_from_id("28") == "WAS"
    assert espn_team_abbr_from_id("14") == "LA"
    assert espn_team_abbr_from_id(33) == "BAL"
    assert espn_team_abbr_from_id(None) is None
    assert espn_team_abbr_from_id("999") is None


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("display", "seconds"),
    [
        ("5:06", 306),
        ("05:06", 306),
        ("15:00", 900),
        ("0:04", 4),
        ("0:04.5", 4),
        ("", None),
        (None, None),
    ],
)
def test_clock_to_seconds(display: str | None, seconds: int | None) -> None:
    assert clock_to_seconds(display) == seconds


def test_seconds_to_clock() -> None:
    assert seconds_to_clock(306) == "5:06"
    assert seconds_to_clock(0) == "0:00"
    assert seconds_to_clock(None) == "--:--"


def test_half_and_game_seconds() -> None:
    assert half_seconds_remaining(1, 900) == 1800
    assert half_seconds_remaining(2, 300) == 300
    assert half_seconds_remaining(3, 900) == 1800
    assert game_seconds_remaining(4, 300) == 300
    assert game_seconds_remaining(1, 900) == 3600
    # Overtime collapses to the period clock, matching nflverse.
    assert game_seconds_remaining(5, 480) == 480
    assert half_seconds_remaining(5, 480) == 480


# --------------------------------------------------------------------------- #
# --at
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "quarter", "seconds"),
    [
        ("Q4 05:00", 4, 300),
        ("q4 5:00", 4, 300),
        ("4 05:00", 4, 300),
        ("Q1 15:00", 1, 900),
        ("Q4", 4, 900),
        ("OT 10:00", 5, 600),
        ("OT", 5, 600),
        ("OT2 5:00", 6, 300),
    ],
)
def test_parse_at(text: str, quarter: int, seconds: int) -> None:
    at = parse_at(text)
    assert at == AtTime(quarter=quarter, clock_seconds=seconds)


def test_parse_at_passthrough_and_none() -> None:
    assert parse_at(None) is None
    at = AtTime(quarter=2, clock_seconds=120)
    assert parse_at(at) is at


def test_parse_at_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        parse_at("late in the fourth")


def test_at_order_key_is_monotonic_in_game_time() -> None:
    assert parse_at("Q1 15:00").order_key < parse_at("Q1 00:01").order_key
    assert parse_at("Q1 00:01").order_key < parse_at("Q2 15:00").order_key
    assert parse_at("Q4 00:00").order_key < parse_at("OT 10:00").order_key


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def _play(period: int, clock: int, **kwargs: object) -> LivePlay:
    defaults: dict[str, object] = {
        "play_id": f"{period}-{clock}",
        "period": period,
        "clock_seconds": clock,
        "clock_display": seconds_to_clock(clock),
        "team": "BAL",
        "home_score": 0,
        "away_score": 0,
    }
    defaults.update(kwargs)
    return LivePlay(**defaults)  # type: ignore[arg-type]


def _snapshot(drives: list[LiveDrive], **kwargs: object) -> LiveSnapshot:
    defaults: dict[str, object] = {
        "source": "test",
        "fetched_at": datetime(2024, 9, 6, 4, 0, tzinfo=UTC),
        "home_team": "KC",
        "away_team": "BAL",
        "state": "post",
        "completed": True,
        "period": 4,
        "clock_seconds": 0,
        "home_score": 27,
        "away_score": 20,
        "drives": drives,
    }
    defaults.update(kwargs)
    return LiveSnapshot(**defaults)  # type: ignore[arg-type]


def test_truncate_reports_the_requested_clock_not_the_last_play() -> None:
    drives = [
        LiveDrive(
            drive_number=1,
            team="BAL",
            result="Field goal",
            plays=[
                _play(4, 400, home_score=27, away_score=17),
                _play(4, 306, home_score=27, away_score=17),
                _play(
                    4,
                    294,
                    home_score=27,
                    away_score=20,
                    start_down=4,
                    start_distance=4,
                    start_yards_to_endzone=14,
                ),
            ],
        )
    ]
    truncated = truncate_snapshot(_snapshot(drives), parse_at("Q4 05:00"))

    assert (truncated.period, truncated.clock_seconds) == (4, 300)
    assert truncated.clock_display == "5:00"
    # Score is the last *completed* play's post-play score: the field goal at
    # 4:54 has not happened yet.
    assert (truncated.home_score, truncated.away_score) == (27, 17)
    # Situation is the pre-snap state of the next play.
    assert (truncated.down, truncated.distance, truncated.yards_to_endzone) == (4, 4, 14)
    assert truncated.possession == "BAL"
    assert truncated.posteam_is_home is False
    assert truncated.play_count == 2
    assert truncated.at == parse_at("Q4 05:00")


def test_truncate_flips_a_final_game_back_to_live() -> None:
    drives = [LiveDrive(drive_number=1, plays=[_play(1, 800), _play(4, 100)])]
    truncated = truncate_snapshot(_snapshot(drives), parse_at("Q2 05:00"))
    assert truncated.state == "in"
    assert truncated.completed is False
    assert truncated.status_detail == "Q2 5:00"


def test_truncate_past_the_end_keeps_the_final_state() -> None:
    drives = [LiveDrive(drive_number=1, result="Touchdown", plays=[_play(1, 800), _play(4, 100)])]
    snapshot = _snapshot(drives)
    truncated = truncate_snapshot(snapshot, parse_at("OT 10:00"))
    assert truncated.state == "post"
    assert truncated.completed is True
    assert truncated.play_count == 2
    assert truncated.drives[0].result == "Touchdown"


def test_truncate_blanks_the_result_of_the_drive_in_progress() -> None:
    """The eventual drive result is an outcome of plays that have not happened."""
    drives = [
        LiveDrive(
            drive_number=1,
            team="BAL",
            result="Field goal",
            display_result="Field Goal",
            description="13 plays, 56 yards, 5:31",
            yards=56,
            offensive_plays=13,
            is_score=True,
            plays=[_play(4, 400), _play(4, 306), _play(4, 294, scoring_play=True)],
        )
    ]
    drive = truncate_snapshot(_snapshot(drives), parse_at("Q4 05:00")).drives[0]
    assert drive.result is None
    assert drive.display_result is None
    assert drive.description is None
    assert drive.yards is None
    assert drive.is_score is False
    assert drive.offensive_plays == 2


def test_truncate_drops_a_drive_that_had_not_started() -> None:
    drives = [
        LiveDrive(drive_number=1, plays=[_play(1, 800)]),
        LiveDrive(drive_number=2, plays=[_play(4, 100)]),
    ]
    truncated = truncate_snapshot(_snapshot(drives), parse_at("Q2 05:00"))
    assert [d.drive_number for d in truncated.drives] == [1]


def test_truncate_uses_per_play_timeouts_when_the_feed_has_them() -> None:
    drives = [
        LiveDrive(
            drive_number=1,
            plays=[
                _play(4, 400),
                _play(4, 294, home_timeouts_remaining=3, away_timeouts_remaining=1),
            ],
        )
    ]
    truncated = truncate_snapshot(
        _snapshot(drives, home_timeouts=0, away_timeouts=0), parse_at("Q4 05:00")
    )
    assert (truncated.home_timeouts, truncated.away_timeouts) == (3, 1)
    assert truncated.timeouts_exact is True


def test_truncate_marks_timeouts_inexact_when_the_feed_has_none() -> None:
    drives = [LiveDrive(drive_number=1, plays=[_play(4, 400), _play(4, 294)])]
    truncated = truncate_snapshot(
        _snapshot(drives, home_timeouts=2, away_timeouts=0), parse_at("Q4 05:00")
    )
    assert (truncated.home_timeouts, truncated.away_timeouts) == (2, 0)
    assert truncated.timeouts_exact is False


def test_truncate_hides_injuries_reported_after_the_cutoff() -> None:
    from nfl.data.live.base import LiveInjury

    wallclock = datetime(2024, 9, 6, 3, 27, tzinfo=UTC)
    drives = [
        LiveDrive(
            drive_number=1,
            plays=[
                _play(4, 306, wallclock=wallclock),
                _play(4, 294, wallclock=wallclock + timedelta(minutes=1)),
            ],
        )
    ]
    injuries = [
        LiveInjury(
            team="BAL", player_name="Early Report", reported_at=wallclock - timedelta(hours=2)
        ),
        LiveInjury(
            team="BAL", player_name="Later Report", reported_at=wallclock + timedelta(minutes=30)
        ),
        LiveInjury(team="KC", player_name="No Timestamp"),
    ]
    truncated = truncate_snapshot(_snapshot(drives, injuries=injuries), parse_at("Q4 05:00"))
    assert [i.player_name for i in truncated.injuries] == ["Early Report", "No Timestamp"]


# --------------------------------------------------------------------------- #
# Snapshot conveniences
# --------------------------------------------------------------------------- #


def test_staleness_uses_the_play_wallclock() -> None:
    wallclock = datetime(2024, 9, 6, 3, 27, tzinfo=UTC)
    snapshot = _snapshot(
        [LiveDrive(drive_number=1, plays=[_play(4, 306, wallclock=wallclock)])], state="in"
    )
    now = wallclock + timedelta(minutes=5)
    assert snapshot.staleness(now) == timedelta(minutes=5)
    assert snapshot.is_stale(timedelta(minutes=2), now=now) is True
    assert snapshot.is_stale(timedelta(minutes=10), now=now) is False


def test_a_finished_game_is_never_stale() -> None:
    wallclock = datetime(2024, 9, 6, 3, 27, tzinfo=UTC)
    snapshot = _snapshot([LiveDrive(drive_number=1, plays=[_play(4, 306, wallclock=wallclock)])])
    assert snapshot.state == "post"
    assert snapshot.is_stale(now=wallclock + timedelta(days=400)) is False


def test_staleness_is_none_without_a_wallclock() -> None:
    snapshot = _snapshot([LiveDrive(drive_number=1, plays=[_play(4, 306)])], state="in")
    assert snapshot.staleness() is None
    assert snapshot.is_stale() is False


def test_plays_flattens_drives_in_order() -> None:
    snapshot = _snapshot(
        [
            LiveDrive(drive_number=1, plays=[_play(1, 900), _play(1, 850)]),
            LiveDrive(drive_number=2, plays=[_play(1, 800)]),
        ]
    )
    assert [p.clock_seconds for p in snapshot.plays] == [900, 850, 800]
    assert snapshot.play_count == 3
    assert snapshot.last_play is not None
    assert snapshot.last_play.clock_seconds == 800


# --------------------------------------------------------------------------- #
# Crosswalk
# --------------------------------------------------------------------------- #


def test_resolve_espn_event_id_from_a_frame(schedules: pl.DataFrame) -> None:
    assert resolve_espn_event_id("2024_01_BAL_KC", schedules) == "401671789"


def test_resolve_espn_event_id_from_a_parquet_path(schedules_path) -> None:  # noqa: ANN001
    assert resolve_espn_event_id("2024_01_BAL_KC", schedules_path) == "401671789"


def test_resolve_espn_event_id_from_a_directory(schedules_path) -> None:  # noqa: ANN001
    assert resolve_espn_event_id("2024_01_BAL_KC", schedules_path.parent) == "401671789"


def test_resolve_espn_event_id_unknown_game(schedules: pl.DataFrame) -> None:
    with pytest.raises(KeyError, match="not in schedules"):
        resolve_espn_event_id("1999_01_XXX_YYY", schedules)


def test_resolve_espn_event_id_when_the_column_is_empty(schedules: pl.DataFrame) -> None:
    blank = schedules.with_columns(pl.lit(None, dtype=pl.Utf8).alias("espn"))
    with pytest.raises(KeyError, match="no espn id"):
        resolve_espn_event_id("2024_01_BAL_KC", blank)


def test_team_record_as_of_only_counts_prior_weeks(schedules: pl.DataFrame) -> None:
    assert team_record_as_of("KC", 2024, 1, schedules).overall == "0-0"
    week5 = team_record_as_of("KC", 2024, 5, schedules)
    assert week5.wins == 4
    assert week5.losses == 0
    assert week5.overall == "4-0"
