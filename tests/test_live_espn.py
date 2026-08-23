"""ESPN adapter, driven entirely by recorded payloads through respx."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from nfl.data.live.base import GameRef
from nfl.data.live.espn import CORE_BASE, SITE_BASE, EspnLiveFeed, parse_weather

EVENT_ID = "401671789"
GAME_ID = "2024_01_BAL_KC"


@pytest.fixture
def feed(schedules_path: Path) -> EspnLiveFeed:
    # backoff=0 keeps the retry test instant.
    with EspnLiveFeed(schedules=schedules_path, backoff=0.0) as f:
        yield f


def _mock_game(
    mock: respx.MockRouter,
    summary: dict[str, Any],
    situation: dict[str, Any] | None,
    scoreboard: dict[str, Any] | None = None,
) -> None:
    mock.get(url__startswith=f"{SITE_BASE}/summary").respond(json=summary)
    route = mock.get(url__startswith=f"{CORE_BASE}/events/{EVENT_ID}")
    if situation is None:
        route.respond(status_code=404)
    else:
        route.respond(json=situation)
    if scoreboard is not None:
        mock.get(url__startswith=f"{SITE_BASE}/scoreboard").respond(json=scoreboard)


# --------------------------------------------------------------------------- #
# list_games
# --------------------------------------------------------------------------- #


def test_list_games_from_a_completed_slate(
    feed: EspnLiveFeed, espn_scoreboard: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/scoreboard").respond(json=espn_scoreboard)
        games = feed.list_games(date(2024, 9, 5))

    assert len(games) == 1
    game = games[0]
    assert game.espn_event_id == EVENT_ID
    assert game.nflverse_game_id == GAME_ID
    assert (game.home_team, game.away_team) == ("KC", "BAL")
    assert game.season == 2024
    assert game.season_type == "REG"
    assert game.week == 1
    assert game.state == "post"
    assert game.completed is True
    assert game.kickoff_utc == datetime(2024, 9, 6, 0, 40, tzinfo=UTC)
    assert game.venue_name


def test_list_games_sends_the_date_as_yyyymmdd(
    feed: EspnLiveFeed, espn_scoreboard: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{SITE_BASE}/scoreboard").respond(json=espn_scoreboard)
        feed.list_games(date(2024, 9, 5))
    assert route.calls.last.request.url.params["dates"] == "20240905"


def test_list_games_on_a_live_slate(
    feed: EspnLiveFeed, espn_scoreboard_live: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/scoreboard").respond(json=espn_scoreboard_live)
        games = feed.list_games(date(2026, 8, 22))
    assert games and games[0].state == "in"
    assert games[0].completed is False
    # No 2026 row in the 2024 schedules fixture: the reverse crosswalk must not raise.
    assert games[0].nflverse_game_id is None


# --------------------------------------------------------------------------- #
# snapshot
# --------------------------------------------------------------------------- #


def test_snapshot_of_a_completed_game(
    feed: EspnLiveFeed, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        _mock_game(mock, espn_summary, espn_situation)
        snapshot = feed.snapshot(GAME_ID)

    assert snapshot.source == "espn"
    assert snapshot.game_id == GAME_ID
    assert snapshot.espn_event_id == EVENT_ID
    assert (snapshot.home_team, snapshot.away_team) == ("KC", "BAL")
    assert (snapshot.home_score, snapshot.away_score) == (27, 20)
    assert snapshot.state == "post"
    assert snapshot.completed is True
    assert snapshot.season == 2024
    assert snapshot.week == 1
    # Timeouts come only from the core situation endpoint.
    assert (snapshot.home_timeouts, snapshot.away_timeouts) == (2, 0)
    assert snapshot.timeouts_exact is True
    assert snapshot.venue is not None and snapshot.venue.name == "Arrowhead Stadium"
    assert snapshot.home_record is not None and snapshot.home_record.overall == "1-0"
    assert snapshot.away_record is not None and snapshot.away_record.overall == "0-1"


def test_snapshot_flattens_drives_into_plays_with_wallclocks(
    feed: EspnLiveFeed, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        _mock_game(mock, espn_summary, espn_situation)
        snapshot = feed.snapshot(GAME_ID)

    assert len(snapshot.drives) == 20
    assert snapshot.play_count == 188
    assert all(d.plays for d in snapshot.drives)
    assert snapshot.last_play_wallclock == datetime(2024, 9, 6, 3, 52, 59, tzinfo=UTC)
    # Every play carries the fields the replay feed also fills.
    first = snapshot.plays[0]
    assert first.period == 1
    assert first.clock_seconds == 900
    assert first.text
    # ESPN publishes no per-play timeouts. Documented asymmetry, asserted so it
    # cannot change silently.
    assert all(p.home_timeouts_remaining is None for p in snapshot.plays)


def test_snapshot_truncates_to_a_requested_clock(
    feed: EspnLiveFeed, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        _mock_game(mock, espn_summary, espn_situation)
        snapshot = feed.snapshot(GAME_ID, at="Q4 05:00")

    assert (snapshot.period, snapshot.clock_seconds) == (4, 300)
    assert (snapshot.home_score, snapshot.away_score) == (27, 17)
    assert snapshot.possession == "BAL"
    assert (snapshot.down, snapshot.distance, snapshot.yards_to_endzone) == (4, 4, 14)
    assert snapshot.state == "in"
    assert snapshot.completed is False
    # ESPN can only ever report *current* timeouts.
    assert snapshot.timeouts_exact is False


def test_snapshot_survives_a_missing_core_situation(
    feed: EspnLiveFeed, espn_summary: dict[str, Any]
) -> None:
    """404 on the situation endpoint is normal for some games; not an error."""
    with respx.mock(assert_all_called=False) as mock:
        _mock_game(mock, espn_summary, None)
        snapshot = feed.snapshot(GAME_ID)

    assert snapshot.home_timeouts is None
    assert snapshot.away_timeouts is None
    assert snapshot.timeouts_exact is False
    # Possession still resolves, from the last play.
    assert snapshot.possession in ("KC", "BAL")


def test_snapshot_accepts_a_game_ref_with_only_an_event_id(
    feed: EspnLiveFeed, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        _mock_game(mock, espn_summary, espn_situation)
        snapshot = feed.snapshot(GameRef(espn_event_id=EVENT_ID))
    assert snapshot.espn_event_id == EVENT_ID
    assert snapshot.game_id == GAME_ID  # filled by the reverse crosswalk


def test_snapshot_requests_the_resolved_event_id(
    feed: EspnLiveFeed, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{SITE_BASE}/summary").respond(json=espn_summary)
        mock.get(url__startswith=f"{CORE_BASE}/events/{EVENT_ID}").respond(json=espn_situation)
        feed.snapshot(GAME_ID)
    assert route.calls.last.request.url.params["event"] == EVENT_ID


def test_missing_summary_raises(feed: EspnLiveFeed) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/summary").respond(status_code=404)
        with pytest.raises(KeyError, match="no summary"):
            feed.snapshot(GAME_ID)


# --------------------------------------------------------------------------- #
# HTTP behaviour
# --------------------------------------------------------------------------- #


def test_transient_server_errors_are_retried(
    feed: EspnLiveFeed, espn_scoreboard: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/scoreboard").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(429),
                httpx.Response(200, json=espn_scoreboard),
            ]
        )
        games = feed.list_games(date(2024, 9, 5))
    assert len(games) == 1


def test_retries_are_bounded(feed: EspnLiveFeed) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{SITE_BASE}/scoreboard").respond(status_code=503)
        with pytest.raises(httpx.HTTPStatusError):
            feed.list_games(date(2024, 9, 5))
    assert route.call_count == 3


def test_transport_errors_are_retried_then_raised(feed: EspnLiveFeed) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{SITE_BASE}/scoreboard").mock(
            side_effect=httpx.ConnectError("boom")
        )
        with pytest.raises(httpx.ConnectError):
            feed.list_games(date(2024, 9, 5))
    assert route.call_count == 3


def test_requests_carry_polite_headers(feed: EspnLiveFeed, espn_scoreboard: dict[str, Any]) -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url__startswith=f"{SITE_BASE}/scoreboard").respond(json=espn_scoreboard)
        feed.list_games(date(2024, 9, 5))
    headers = route.calls.last.request.headers
    assert "nfl-winprob" in headers["user-agent"]
    assert headers["accept"] == "application/json"


# --------------------------------------------------------------------------- #
# Weather quirk
# --------------------------------------------------------------------------- #


def test_weather_unswaps_espns_transposed_fields() -> None:
    weather = parse_weather(
        {
            "displayValue": "2",
            "conditionId": "Mostly sunny",
            "temperature": 82,
            "highTemperature": 82,
        }
    )
    assert weather is not None
    assert weather.display_value == "Mostly sunny"
    assert weather.condition_id == "2"
    assert weather.temperature_f == 82


def test_weather_leaves_a_sane_payload_alone() -> None:
    weather = parse_weather({"displayValue": "Cloudy", "conditionId": "2", "temperature": 52})
    assert weather is not None
    assert weather.display_value == "Cloudy"
    assert weather.condition_id == "2"


def test_weather_is_none_when_absent() -> None:
    assert parse_weather(None) is None
    assert parse_weather({}) is None
